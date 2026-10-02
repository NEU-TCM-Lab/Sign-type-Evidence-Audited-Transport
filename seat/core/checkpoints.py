"""Save and restore the complete readout definition, including non-tensor settings."""
from __future__ import annotations

import json
from pathlib import Path
import torch
from common import LABELS, RUNS_DIR, FEATURES_DIR
from data_validation import alignment_order
from ot_readout_head import AttrReadoutHead
from ot_sign_head import SignOTHead
from ot_gen_head import OTGenHead, DualOTHead


def model_config_from_args(args, feature_dim: int, query_emb=None) -> dict:
    kwargs = dict(feature_dim=feature_dim, num_labels=len(LABELS), proj_dim=args.proj_dim,
                  eps_init=args.eps_init, sinkhorn_iters=args.sinkhorn_iters,
                  learn_eps=not args.fix_eps, dropout=args.dropout)
    if args.readout == "signot":
        kwargs.update(branch=args.branch, evidence=args.evidence, ot_k=args.ot_k,
                      use_mask_mass=args.use_mask_mass, beta_init=args.beta_init,
                      beta_per_class=not args.beta_scalar, proto_div=args.proto_div_lambda > 0,
                      ot_demand=args.ot_demand, ot_relax=args.ot_relax, uot_rho=args.uot_rho,
                      partial_m=args.partial_m, partial_tau=args.partial_tau, mass_gate=args.mass_gate)
    elif args.readout in {"otgen", "dualot"}:
        kwargs.update(ot_k=args.ot_k, lam_init=args.lam_init, per_class_lam=args.per_class_lam,
                      combine=args.otgen_combine, peak_topk=args.peak_topk)
        if args.readout == "dualot":
            kwargs["dino_dim"] = args.dino_dim
    else:
        kwargs.update(readout=args.readout, mask_guided=not args.no_mask_guided,
                      uot_rho=args.uot_rho, num_heads=args.num_heads,
                      ot_reg=args.ot_reg_lambda > 0, ot_reg_eps=args.ot_reg_eps,
                      spatial_reg=args.spatial_lambda > 0)
    return {"version": 2, "readout": args.readout, "kwargs": kwargs,
            "use_text_query": query_emb is not None}


def build_readout_model(config: dict, query_emb=None):
    if config.get("version") != 2:
        raise ValueError("Unsupported model config version; a complete version 2 configuration is required")
    ro = config["readout"]
    if ro not in {"signot", "otgen", "dualot", "mlp", "softmax", "ot", "mpsa", "uot"}:
        raise ValueError("Unknown readout in model configuration")
    constructors = {"signot": SignOTHead, "otgen": OTGenHead, "dualot": DualOTHead}
    constructor = constructors.get(ro, AttrReadoutHead)
    kwargs = dict(config["kwargs"])
    if kwargs.get("num_labels") != len(LABELS) or ("readout" in kwargs and kwargs["readout"] != ro):
        raise ValueError("Model configuration has inconsistent readout or label count")
    # These values change the forward pass without changing tensor shapes.
    required = {"feature_dim", "num_labels", "proj_dim", "sinkhorn_iters"}
    if ro in {"otgen", "dualot"}:
        required |= {"peak_topk", "combine", "per_class_lam"}
    elif ro == "signot":
        required |= {"branch", "evidence", "ot_demand", "ot_relax", "uot_rho", "partial_m", "partial_tau", "mass_gate"}
    else:
        required |= {"num_heads", "mask_guided", "uot_rho", "readout"}
    if required - kwargs.keys():
        raise ValueError(f"Incomplete model configuration: {sorted(required - kwargs.keys())}")
    if config.get("use_text_query"):
        if query_emb is None or constructor is not AttrReadoutHead:
            raise ValueError("Text-query embeddings are missing or incompatible with this readout")
        kwargs["query_emb"] = query_emb
    return constructor(**kwargs)


def load_trained_model(run: str, device, expected_feature_dim: int | None = None):
    run_dir = RUNS_DIR / run
    checkpoint = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    if list(checkpoint["label_order"]) != LABELS:
        raise ValueError("Checkpoint label order mismatch")
    config = checkpoint.get("model_config")
    if config is None and (run_dir / "model_config.json").is_file():
        config = json.loads((run_dir / "model_config.json").read_text(encoding="utf-8"))
    if config is None:
        raise ValueError("Legacy checkpoint lacks forward-pass settings. Retrain or supply a verified "
                         "model_config.json; head count and peak_topk cannot be inferred from weights.")
    if expected_feature_dim is not None and config["kwargs"]["feature_dim"] != expected_feature_dim:
        raise ValueError("Checkpoint and input feature dimensions differ")
    state = checkpoint["model_state"]
    model = build_readout_model(config, query_emb=state.get("query_emb"))
    model.load_state_dict(state, strict=True)
    return model.to(device).eval(), config


def load_run_payload(run: str, tag: str, split: str, features_dir: Path = FEATURES_DIR) -> dict:
    run_dir = RUNS_DIR / run
    metadata_path = run_dir / "input_config.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    if (run_dir / "best.pt").exists():
        saved = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False).get("input_config")
        if saved is not None:
            if metadata and metadata != saved:
                raise ValueError("Checkpoint and sidecar input configurations differ")
            metadata = saved
    if not metadata:
        raise ValueError("Run lacks a verified input_config.json or checkpoint input configuration")
    if metadata.get("tag", tag) != tag:
        raise ValueError("Requested cache tag differs from the trained model's input tag")
    payload = torch.load(features_dir / f"{split}_patchgrid_{tag}.pt", map_location="cpu", weights_only=False)
    if metadata.get("geometry", payload.get("geometry", "square")) != payload.get("geometry", "square"):
        raise ValueError("Cache geometry differs from the trained model's input geometry")
    extra = metadata.get("extra_tag")
    if extra:
        other = torch.load(features_dir / f"{split}_patchgrid_{extra}.pt", map_location="cpu", weights_only=False)
        order = alignment_order(payload, other)
        if other.get("geometry", "square") != payload.get("geometry", "square"):
            raise ValueError("Concatenated feature grids have different geometry")
        if payload["mask_weights"].shape != other["mask_weights"][order].shape or not torch.allclose(
                payload["mask_weights"].float(), other["mask_weights"][order].float(), atol=1e-3, rtol=1e-3):
            raise ValueError("Additional cache uses a different mask grid")
        payload["patch_features"] = torch.cat([payload["patch_features"], other["patch_features"][order]], dim=-1)
        payload["feature_dim"] = int(payload["patch_features"].shape[-1])
    return payload
