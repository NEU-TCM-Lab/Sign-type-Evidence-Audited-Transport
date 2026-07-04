from __future__ import annotations

# Train + eval an AttrReadoutHead (mlp | softmax | ot) over a frozen patch-grid cache.
# Same data / split / protocol as the maskpool baseline -> isolates the READOUT mechanism.

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from common import FEATURES_DIR, LABELS, RUNS_DIR, assert_label_order, set_seed, timestamp, write_json
from metrics import compute_metrics, find_best_thresholds, per_class_rows, sigmoid_np
from ot_readout_head import AttrReadoutHead
from ot_sign_head import SignOTHead
from ot_gen_head import OTGenHead, DualOTHead
from losses import LogitAdjuster, build_loss, compute_pos_prior, loss_uses_logit_adjustment


def grid_path(split: str, tag: str, features_dir: Path) -> Path:
    return Path(features_dir) / f"{split}_patchgrid_{tag}.pt"


class GridDataset(Dataset):
    def __init__(self, payload: dict, limit: int | None = None) -> None:
        assert_label_order(payload["label_order"])
        n = payload["labels"].shape[0] if limit is None else min(limit, payload["labels"].shape[0])
        self.pf = payload["patch_features"][:n]  # [N,T,D] fp16
        self.mw = payload["mask_weights"][:n]    # [N,T] fp16
        self.y = payload["labels"][:n].float()   # [N,8]

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, i):
        return self.pf[i], self.mw[i], self.y[i]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True, help="patchgrid cache tag, e.g. siglip2grid_seed42")
    p.add_argument("--readout", required=True, choices=["mlp", "softmax", "ot", "mpsa", "uot", "signot", "otgen", "dualot"])
    p.add_argument("--dino-dim", type=int, default=1024, help="dualot: split index between DINOv2 and color features.")
    # --- OTGenHead (OT-generalized attention readout: semi-relaxed OT, lam=0 == attention baseline) ---
    p.add_argument("--lam-init", type=float, default=0.02, help="otgen initial patch-competition (sigmoid logit).")
    p.add_argument("--per-class-lam", dest="per_class_lam", action="store_true", default=True)
    p.add_argument("--no-per-class-lam", dest="per_class_lam", action="store_false")
    p.add_argument("--otgen-combine", choices=["mean", "concat"], default="mean", help="otgen: combine K prototype vectors (concat = OTKE-style).")
    p.add_argument("--peak-topk", type=int, default=0, help="otgen: add per-class top-k peak-similarity detection term (localized rare signs).")
    # --- SignOTHead (two-branch: global classifier + per-label prototype-to-patch OT) ---
    p.add_argument("--proj-dim", type=int, default=256)
    p.add_argument("--branch", choices=["global", "ot", "both"], default="both")
    p.add_argument("--evidence", choices=["ot", "attention", "aot"], default="ot")
    p.add_argument("--ot-demand", choices=["uniform", "mask", "attention"], default="attention",
                   help="aot: OT column/patch demand source (default attention = global-branch saliency).")
    p.add_argument("--ot-relax", choices=["balanced", "unbalanced", "partial"], default="unbalanced",
                   help="aot: OT relaxation (unbalanced KL-rho / partial mass-reservoir / balanced).")
    p.add_argument("--partial-m", type=float, default=0.7, help="aot partial-OT transported mass fraction.")
    p.add_argument("--partial-tau", type=float, default=0.5, help="aot partial-OT dummy-reservoir cost.")
    p.add_argument("--mass-gate", dest="mass_gate", action="store_true", default=True,
                   help="aot: gate ot_logit by per-sign transported mass (default on).")
    p.add_argument("--no-mass-gate", dest="mass_gate", action="store_false", help="aot: disable mass gate.")
    p.add_argument("--ot-k", type=int, default=4, help="prototypes per label (signot).")
    p.add_argument("--beta-init", type=float, default=0.1)
    p.add_argument("--beta-scalar", action="store_true", help="single shared beta instead of per-class.")
    p.add_argument("--proto-div-lambda", type=float, default=0.0, help="prototype-diversity regularizer weight (signot).")
    p.add_argument("--use-mask-mass", action="store_true", help="use tongue mask as Sinkhorn column marginal.")
    p.add_argument("--init-global", type=str, default=None, help="run-name of a global-only signot run; load its global-branch weights.")
    p.add_argument("--freeze-global", action="store_true", help="freeze the global branch; train only OT branch + beta.")
    p.add_argument("--freeze-frontend", action="store_true", help="also load+freeze shared proj/in_norm so the global path == baseline EXACTLY (clean fairness test).")
    p.add_argument("--uot-rho", type=float, default=0.1, help="UOT marginal-relaxation strength.")
    p.add_argument("--ot-reg-lambda", type=float, default=0.0, help="softmax + OT-align aux regularizer weight (V10 attn_align).")
    p.add_argument("--ot-reg-eps", type=float, default=0.1, help="eps of the OT plan used by the aux regularizer.")
    p.add_argument("--spatial-lambda", type=float, default=0.0, help="spatial-coherence (2D TV) regularizer weight.")
    p.add_argument("--coocc-lambda", type=float, default=0.0, help="label-level co-occurrence OT (Sinkhorn) regularizer weight.")
    p.add_argument("--coocc-eps", type=float, default=0.1)
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--features-dir", type=Path, default=FEATURES_DIR)
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit-train", type=int, default=None, help="use first N train samples (subset comparison).")
    p.add_argument("--no-mask-guided", action="store_true")
    p.add_argument("--eps-init", type=float, default=0.05)
    p.add_argument("--fix-eps", action="store_true", help="freeze eps at --eps-init (scan it instead of learning).")
    p.add_argument("--sinkhorn-iters", type=int, default=20)
    p.add_argument("--num-heads", type=int, default=1, help="multi-head attention/transport.")
    p.add_argument("--query-emb-path", type=str, default=None, help="MLLM text-prototype queries (.pt with 'emb' [A,dim]).")
    p.add_argument("--dump-teacher", type=str, default=None, help="after training, dump train/val/test raw logits+ids (for KD).")
    # --- Stage-1 strong-baseline loss knobs (default = historical pos_weight BCE) ---
    p.add_argument("--loss", choices=["bce", "asl", "logit_adjusted_bce", "asl_logit_adjusted"], default="bce")
    p.add_argument("--asl-gamma-pos", type=float, default=0.0)
    p.add_argument("--asl-gamma-neg", type=float, default=4.0)
    p.add_argument("--asl-clip", type=float, default=0.05)
    p.add_argument("--la-tau", type=float, default=1.0)
    pos_grp = p.add_mutually_exclusive_group()
    pos_grp.add_argument("--pos-weight", dest="pos_weight", action="store_true")
    pos_grp.add_argument("--no-pos-weight", dest="pos_weight", action="store_false")
    p.set_defaults(pos_weight=True)
    p.add_argument("--selection-metric", choices=["macro_auroc", "macro_f1", "macro_ap", "rare2", "pale", "ecchy"], default="macro_auroc")
    p.add_argument("--threshold-step", type=float, default=0.01)
    p.add_argument("--experiment-name", type=str, default=None, help="label for per_class_results.csv rows.")
    p.add_argument("--per-class-csv", type=str, default=None, help="append per-class test rows here.")
    p.add_argument("--extra-tag", type=str, default=None, help="concat a second patchgrid (e.g. color_seed42) onto patch_features, aligned by id.")
    return p.parse_args()


def _sel_score(vmet: dict, which: str) -> float:
    if which in ("rare2", "pale", "ecchy"):   # select for the weak rare signs
        pl = vmet["per_label"]
        if which == "pale":
            return float(pl["TonguePale"]["f1"])
        if which == "ecchy":
            return float(pl["Ecchymosis"]["f1"])
        return float(np.mean([pl["TonguePale"]["f1"], pl["Ecchymosis"]["f1"]]))
    key = {"macro_auroc": "auroc", "macro_f1": "f1", "macro_ap": "ap"}[which]
    s = vmet["macro"][key]
    if s is None or not np.isfinite(s):
        s = vmet["macro"]["ap"] or vmet["macro"]["f1"] or 0.0
    return float(s)


def cooccurrence_cost(Y: torch.Tensor) -> torch.Tensor:
    # 8x8 ground cost = 1 - Jaccard(sign_i, sign_j); co-occurring signs are "close".
    K = Y.shape[1]
    C = torch.ones(K, K)
    for i in range(K):
        for j in range(K):
            inter = ((Y[:, i] == 1) & (Y[:, j] == 1)).sum().float()
            union = ((Y[:, i] == 1) | (Y[:, j] == 1)).sum().float().clamp(min=1)
            C[i, j] = 1.0 - inter / union
    return C  # [K,K], diag 0


def ot_transport_cost(a: torch.Tensor, b: torch.Tensor, C: torch.Tensor, eps: float, iters: int = 50) -> torch.Tensor:
    # entropic OT transport cost <P,C> between histograms a,b (differentiable in a). Small K, cheap.
    a = a.clamp(min=1e-6); a = a / a.sum()
    b = b.clamp(min=1e-6); b = b / b.sum()
    Kmat = torch.exp(-C / eps)
    u = torch.ones_like(a)
    for _ in range(iters):
        v = b / (Kmat.t() @ u).clamp(min=1e-9)
        u = a / (Kmat @ v).clamp(min=1e-9)
    P = u.unsqueeze(1) * Kmat * v.unsqueeze(0)
    return (P * C).sum()


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    logits_all, y_all = [], []
    for pf, mw, y in loader:
        out = model(pf.to(device), mw.to(device))
        logits_all.append(out.float().cpu().numpy())
        y_all.append(y.numpy())
    return np.concatenate(logits_all, 0), np.concatenate(y_all, 0)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    run_name = args.run_name or f"readout_{args.readout}_{timestamp()}"
    run_dir = RUNS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_pl = torch.load(grid_path("train", args.tag, args.features_dir), map_location="cpu", weights_only=False)
    val_pl = torch.load(grid_path("val", args.tag, args.features_dir), map_location="cpu", weights_only=False)
    test_pl = torch.load(grid_path("test", args.tag, args.features_dir), map_location="cpu", weights_only=False)
    if args.extra_tag:   # concat a second patchgrid (e.g. color_seed42) onto patch_features, aligned by id
        for pl, sp in [(train_pl, "train"), (val_pl, "val"), (test_pl, "test")]:
            ex = torch.load(grid_path(sp, args.extra_tag, args.features_dir), map_location="cpu", weights_only=False)
            eidx = {str(x): i for i, x in enumerate(ex["ids"])}
            order = [eidx[str(b)] for b in pl["ids"]]
            ef = ex["patch_features"][order]                       # [N,T,Ce] aligned to base order
            pl["patch_features"] = torch.cat([pl["patch_features"], ef], dim=-1)
            pl["feature_dim"] = int(pl["patch_features"].shape[-1])
        print(f"[extra-tag {args.extra_tag}] concatenated -> feature_dim={train_pl['feature_dim']}")
    feature_dim = int(train_pl["feature_dim"])

    train_ds = GridDataset(train_pl, limit=args.limit_train)
    val_ds, test_ds = GridDataset(val_pl), GridDataset(test_pl)
    print(f"train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} (limit_train={args.limit_train})")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)

    query_emb = None
    if args.query_emb_path:
        qd = torch.load(args.query_emb_path, map_location="cpu", weights_only=False)
        assert list(qd["label_order"]) == LABELS, "query_emb label order mismatch"
        query_emb = qd["emb"]
        print(f"using MLLM text-prototype queries from {args.query_emb_path} shape {tuple(query_emb.shape)}")
    if args.readout == "signot":
        model = SignOTHead(
            feature_dim=feature_dim, num_labels=len(LABELS), proj_dim=args.proj_dim,
            branch=args.branch, evidence=args.evidence, ot_k=args.ot_k,
            eps_init=args.eps_init, sinkhorn_iters=args.sinkhorn_iters, learn_eps=not args.fix_eps,
            use_mask_mass=args.use_mask_mass, beta_init=args.beta_init,
            beta_per_class=not args.beta_scalar, proto_div=(args.proto_div_lambda > 0), dropout=args.dropout,
            ot_demand=args.ot_demand, ot_relax=args.ot_relax, uot_rho=args.uot_rho,
            partial_m=args.partial_m, partial_tau=args.partial_tau, mass_gate=args.mass_gate,
        ).to(device)
        # front-end (shared proj/in_norm) is included when --freeze-frontend, so the global path can be
        # held EXACTLY at the baseline (otherwise these shared params drift under the OT objective).
        frontend = [n for n, _ in model.named_parameters() if n.startswith(("in_norm.", "proj."))]
        if args.init_global:
            src = torch.load(RUNS_DIR / args.init_global / "best.pt", map_location=device)["model_state"]
            gnames = set(model.global_param_names()) | (set(frontend) if args.freeze_frontend else set())
            loaded = model.load_state_dict({k: v for k, v in src.items() if k in gnames}, strict=False)
            print(f"[init-global] loaded {len(gnames)} params from {args.init_global} "
                  f"(incl front-end={args.freeze_frontend}; missing={len(loaded.missing_keys)} unexpected={len(loaded.unexpected_keys)})")
        if args.freeze_global:
            frozen = set(model.global_param_names()) | (set(frontend) if args.freeze_frontend else set())
            for n, prm in model.named_parameters():
                if n in frozen:
                    prm.requires_grad_(False)
            print(f"[freeze-global] froze {len(frozen)} params (front-end={args.freeze_frontend}); training OT+beta only")
    elif args.readout == "otgen":
        model = OTGenHead(
            feature_dim=feature_dim, num_labels=len(LABELS), proj_dim=args.proj_dim,
            ot_k=args.ot_k, eps_init=args.eps_init, sinkhorn_iters=args.sinkhorn_iters,
            learn_eps=not args.fix_eps, lam_init=args.lam_init, per_class_lam=args.per_class_lam,
            combine=args.otgen_combine, peak_topk=args.peak_topk, dropout=args.dropout,
        ).to(device)
    elif args.readout == "dualot":
        model = DualOTHead(
            feature_dim=feature_dim, dino_dim=args.dino_dim, num_labels=len(LABELS), proj_dim=args.proj_dim,
            ot_k=args.ot_k, eps_init=args.eps_init, sinkhorn_iters=args.sinkhorn_iters,
            learn_eps=not args.fix_eps, lam_init=args.lam_init, per_class_lam=args.per_class_lam,
            combine=args.otgen_combine, peak_topk=args.peak_topk, dropout=args.dropout,
        ).to(device)
    else:
        model = AttrReadoutHead(
            feature_dim=feature_dim, num_labels=len(LABELS), readout=args.readout,
            mask_guided=not args.no_mask_guided, eps_init=args.eps_init, learn_eps=not args.fix_eps,
            sinkhorn_iters=args.sinkhorn_iters, uot_rho=args.uot_rho, num_heads=args.num_heads,
            ot_reg=(args.ot_reg_lambda > 0), ot_reg_eps=args.ot_reg_eps,
            spatial_reg=(args.spatial_lambda > 0), query_emb=query_emb, dropout=args.dropout,
        ).to(device)

    pos = train_ds.y.sum(0)
    neg = train_ds.y.shape[0] - pos
    pos_weight = torch.clamp(neg / torch.clamp(pos, min=1.0), min=0.2, max=5.0).to(device) if args.pos_weight else None
    criterion = build_loss(
        args.loss, pos_weight=pos_weight,
        asl_gamma_pos=args.asl_gamma_pos, asl_gamma_neg=args.asl_gamma_neg, asl_clip=args.asl_clip,
    )
    adjuster = None
    if loss_uses_logit_adjustment(args.loss):
        adjuster = LogitAdjuster(compute_pos_prior(train_ds.y), tau=args.la_tau).to(device)
        print(f"logit-adjustment tau={args.la_tau} adj={adjuster.adjustment.cpu().numpy().round(3).tolist()}")
    if args.coocc_lambda > 0:
        coocc_C = cooccurrence_cost(train_ds.y).to(device)
        coocc_b = (train_ds.y.mean(0) + 1e-6).to(device)   # empirical sign-frequency target
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=args.weight_decay)

    best_score, best_epoch, bad = -1e9, -1, 0
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        for pf, mw, y in tqdm(train_loader, desc=f"{args.readout}:{epoch}", leave=False):
            opt.zero_grad(set_to_none=True)
            out = model(pf.to(device), mw.to(device))
            adj_out = adjuster(out) if adjuster is not None else out
            loss = criterion(adj_out, y.to(device))
            if args.coocc_lambda > 0:
                qd = torch.sigmoid(out).mean(0)            # batch mean predicted sign distribution
                loss = loss + args.coocc_lambda * ot_transport_cost(qd, coocc_b, coocc_C, args.coocc_eps)
            if args.ot_reg_lambda > 0:
                loss = loss + args.ot_reg_lambda * model.aux_loss
            if args.spatial_lambda > 0:
                loss = loss + args.spatial_lambda * model.spatial_loss
            if args.proto_div_lambda > 0:
                loss = loss + args.proto_div_lambda * model.proto_div_loss
            loss.backward()
            opt.step()
        vlog, vy = evaluate(model, val_loader, device)
        vprob = sigmoid_np(vlog)
        thr = find_best_thresholds(vy, vprob, LABELS, step=args.threshold_step)
        vmet = compute_metrics(vy, vprob, thr, LABELS)
        score = _sel_score(vmet, args.selection_metric)
        eps_now = float(model.log_eps.exp().detach()) if args.readout in {"ot", "mpsa"} else None
        print(f"epoch={epoch} val_macro_auroc={vmet['macro']['auroc']:.4f} "
              f"val_macro_f1={vmet['macro']['f1']:.4f} val_micro_f1={vmet['micro']['f1']:.4f} "
              f"val_sample_f1={vmet['sample_f1']:.4f} [{args.selection_metric}={score:.4f}] eps={eps_now}")
        if score > best_score:
            best_score, best_epoch, bad = score, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_thr = thr
        else:
            bad += 1
            if bad >= args.patience:
                print(f"early stop @ {epoch}, best={best_epoch}")
                break

    model.load_state_dict(best_state)
    torch.save({"model_state": best_state, "label_order": LABELS, "thresholds": best_thr,
                "readout": args.readout, "loss": args.loss}, run_dir / "best.pt")

    if args.dump_teacher:
        dump = {"label_order": np.array(LABELS)}
        for split, pl in [("train", train_pl), ("val", val_pl), ("test", test_pl)]:
            ld = DataLoader(GridDataset(pl), batch_size=128, shuffle=False, num_workers=2)
            lg, yy = evaluate(model, ld, device)
            dump[f"{split}_logits"] = lg
            dump[f"{split}_labels"] = yy
            dump[f"{split}_ids"] = np.array(list(pl["ids"])[:len(lg)])
        np.savez(args.dump_teacher, **dump)
        print(f"[dump-teacher] -> {args.dump_teacher} | train_logits {dump['train_logits'].shape} "
              f"ids[0]={dump['train_ids'][0]}", flush=True)

    exp_name = args.experiment_name or run_name

    # test metrics at threshold=0.5 AND with val-calibrated per-class thresholds
    tlog, ty = evaluate(model, test_loader, device)
    tprob = sigmoid_np(tlog)
    thr_half = {lab: 0.5 for lab in LABELS}
    tmet_half = compute_metrics(ty, tprob, thr_half, LABELS)
    tmet = compute_metrics(ty, tprob, best_thr, LABELS)   # calibrated (thresholds tuned on val)
    tmet["num_samples"] = int(ty.shape[0])
    write_json(run_dir / "metrics_test.json", tmet)
    write_json(run_dir / "metrics_test_thr0.5.json", tmet_half)
    write_json(run_dir / "thresholds.json", {
        "class_names": LABELS,
        "thresholds": [round(float(best_thr[lab]), 4) for lab in LABELS],
        "thresholds_by_name": {lab: round(float(best_thr[lab]), 4) for lab in LABELS},
        "search_range": [0.05, 0.95, args.threshold_step],
        "selected_by": "per_class_f1_on_validation",
    })
    write_json(run_dir / "summary.json", {
        "readout": args.readout, "tag": args.tag, "loss": args.loss, "experiment_name": exp_name,
        "selection_metric": args.selection_metric, "best_epoch": best_epoch,
        "mask_guided": not args.no_mask_guided,
        "final_eps": float(model.log_eps.exp().detach()) if args.readout in {"ot", "mpsa", "signot", "otgen"} else None,
        "signot": ({"branch": args.branch, "evidence": args.evidence, "ot_k": args.ot_k,
                    "proj_dim": args.proj_dim, "use_mask_mass": args.use_mask_mass,
                    "beta_init": args.beta_init, "beta_per_class": not args.beta_scalar,
                    "proto_div": args.proto_div_lambda > 0, "freeze_global": args.freeze_global,
                    "init_global": args.init_global,
                    "ot_demand": args.ot_demand, "ot_relax": args.ot_relax, "uot_rho": args.uot_rho,
                    "partial_m": args.partial_m, "partial_tau": args.partial_tau,
                    "mass_gate": args.mass_gate} if args.readout == "signot" else None),
        "test_thr0.5": {"macro_f1": tmet_half["macro"]["f1"], "micro_f1": tmet_half["micro"]["f1"],
                        "weighted_f1": tmet_half["weighted_f1"], "sample_f1": tmet_half["sample_f1"]},
        "test_calibrated": {"macro_f1": tmet["macro"]["f1"], "micro_f1": tmet["micro"]["f1"],
                            "weighted_f1": tmet["weighted_f1"], "sample_f1": tmet["sample_f1"],
                            "macro_auroc": tmet["macro"]["auroc"], "macro_ap": tmet["macro"]["ap"]},
        "test_macro": tmet["macro"], "test_micro": tmet["micro"],
    })
    if args.per_class_csv:
        import csv as _csv
        csv_path = Path(args.per_class_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        fields = ["experiment_name", "class_name", "support_pos", "support_neg",
                  "threshold", "precision", "recall", "f1", "auroc", "ap"]
        new = not csv_path.exists()
        with open(csv_path, "a", encoding="utf-8", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=fields)
            if new:
                w.writeheader()
            for r in per_class_rows(exp_name, tmet):
                w.writerow(r)
    m = tmet["macro"]
    print(f"[TEST {args.readout}/{args.loss}] thr0.5 MacroF1={tmet_half['macro']['f1']*100:.2f} | "
          f"calibrated MacroF1={m['f1']*100:.2f} AUROC={m['auroc']*100:.2f} MicroF1={tmet['micro']['f1']*100:.2f} "
          f"sampleF1={tmet['sample_f1']*100:.2f} weightedF1={tmet['weighted_f1']*100:.2f}  -> {run_dir}")


if __name__ == "__main__":
    main()
