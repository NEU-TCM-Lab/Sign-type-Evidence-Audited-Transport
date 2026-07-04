from __future__ import annotations

# Two-branch Sign-Prototype OT readout head (SignOTHead) over a frozen DINOv2 patch-token grid.
#
#   final_logit = global_logit + beta * ot_evidence_logit
#
#   - global branch: mask-guided softmax cross-attention readout (reproduces the strong DINOv2
#                    softmax+BCE baseline = 73.45 calibrated macro-F1).
#   - OT evidence branch: each of the 8 labels has K learnable sign prototypes; PER-LABEL,
#     INDEPENDENT entropic-OT (Sinkhorn) transports the K prototypes onto the N patches. The
#     transport plan's patch-marginal is an interpretable evidence heatmap; pooling patch values
#     by it yields a per-class ot_logit. Because the 8 labels are NOT mutually exclusive, the OT is
#     run independently per label (batched over [B*8, K, N]) -- never a joint 8-way competition.
#
# Lessons from prior OT failures encoded here: OT does NOT replace the global classifier (it is an
# additive evidence head with a small-init learnable beta), and it is not just a patch-alignment loss.

import torch
from torch import nn
import torch.nn.functional as F

from ot_readout_head import sinkhorn_log, sinkhorn_unbalanced, sinkhorn_partial   # REUSE Sinkhorn primitives


class SignOTHead(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        num_labels: int = 8,
        proj_dim: int = 256,
        branch: str = "both",            # global | ot | both
        evidence: str = "ot",            # ot | attention   (only used when OT branch active)
        ot_k: int = 4,                   # prototypes per label
        eps_init: float = 0.05,
        sinkhorn_iters: int = 20,
        learn_eps: bool = True,
        use_mask_mass: bool = True,      # mask_weights as Sinkhorn column (patch) marginal
        beta_init: float = 0.1,
        beta_per_class: bool = True,
        proto_div: bool = False,
        dropout: float = 0.2,
        # ---- attention-guided sparse OT (evidence="aot") ----
        ot_demand: str = "attention",    # uniform | mask | attention  (column/patch demand source)
        ot_relax: str = "unbalanced",    # balanced | unbalanced | partial
        uot_rho: float = 0.1,            # unbalanced KL relaxation strength
        partial_m: float = 0.7,          # partial-OT transported mass fraction
        partial_tau: float = 0.5,        # partial-OT dummy-reservoir cost
        mass_gate: bool = True,          # gate ot_logit by per-sign transported mass (confidence)
    ) -> None:
        super().__init__()
        if branch not in {"global", "ot", "both"}:
            raise ValueError(branch)
        if evidence not in {"ot", "attention", "aot"}:
            raise ValueError(evidence)
        if ot_demand not in {"uniform", "mask", "attention"}:
            raise ValueError(ot_demand)
        if ot_relax not in {"balanced", "unbalanced", "partial"}:
            raise ValueError(ot_relax)
        if evidence == "aot" and ot_demand == "attention" and branch != "both":
            raise ValueError("evidence=aot + ot_demand=attention requires branch=both (needs global attention)")
        self.readout = "signot"
        self.num_labels = num_labels
        self.branch = branch
        self.evidence = evidence
        self.ot_k = int(ot_k)
        self.iters = int(sinkhorn_iters)
        self.use_mask_mass = bool(use_mask_mass)
        self.proto_div = bool(proto_div)
        self.ot_demand = ot_demand
        self.ot_relax = ot_relax
        self.uot_rho = float(uot_rho)
        self.partial_m = float(partial_m)
        self.partial_tau = float(partial_tau)
        self.mass_gate = bool(mass_gate)
        # loss hooks (train_readout.py adds these when their lambdas>0; keep defined to avoid AttributeError)
        self.aux_loss = torch.tensor(0.0)
        self.spatial_loss = torch.tensor(0.0)
        self.proto_div_loss = torch.tensor(0.0)

        self.in_norm = nn.LayerNorm(feature_dim)
        self.proj = nn.Linear(feature_dim, proj_dim)
        self.drop = nn.Dropout(dropout)
        self.log_eps = nn.Parameter(torch.log(torch.tensor(float(eps_init))), requires_grad=learn_eps)

        # ---- global branch (mirror of AttrReadoutHead single-head softmax readout) ----
        if branch in {"global", "both"}:
            self.g_queries = nn.Parameter(torch.randn(num_labels, proj_dim) * 0.02)
            self.g_key = nn.Linear(proj_dim, proj_dim)
            self.g_val = nn.Linear(proj_dim, proj_dim)
            self.g_cls_w = nn.Parameter(torch.randn(num_labels, proj_dim) * 0.02)
            self.g_cls_b = nn.Parameter(torch.zeros(num_labels))
            self.g_out_norm = nn.LayerNorm(proj_dim)

        # ---- OT evidence branch (per-label K prototypes) ----
        if branch in {"ot", "both"}:
            self.protos = nn.Parameter(torch.randn(num_labels, self.ot_k, proj_dim) * 0.02)
            self.ot_key = nn.Linear(proj_dim, proj_dim)
            self.ot_val = nn.Linear(proj_dim, proj_dim)
            self.ot_cls_w = nn.Parameter(torch.randn(num_labels, proj_dim) * 0.02)
            self.ot_cls_b = nn.Parameter(torch.zeros(num_labels))
            self.ot_out_norm = nn.LayerNorm(proj_dim)

        if branch == "both":
            if beta_per_class:
                self.beta = nn.Parameter(torch.full((num_labels,), float(beta_init)))
            else:
                self.beta = nn.Parameter(torch.tensor(float(beta_init)))

    # ---- branch names for selective freezing (--freeze-global) ----
    def global_param_names(self) -> list[str]:
        return [n for n, _ in self.named_parameters()
                if n.startswith(("g_queries", "g_key", "g_val", "g_cls_w", "g_cls_b", "g_out_norm"))]

    def _global_branch(self, x, col_logw):
        q = F.normalize(self.g_queries, dim=-1)                  # [A,d]
        k = F.normalize(self.g_key(x), dim=-1)                   # [B,N,d]
        v = self.g_val(x)                                        # [B,N,d]
        sim = torch.einsum("ad,bnd->ban", q, k)                 # [B,A,N]
        logits = sim / 0.1
        if col_logw is not None:
            logits = logits + col_logw.unsqueeze(1)
        attn = torch.softmax(logits, dim=-1)                    # [B,A,N]
        pooled = torch.einsum("ban,bnd->bad", attn, v)          # [B,A,d]
        pooled = self.drop(self.g_out_norm(pooled))
        return (pooled * self.g_cls_w).sum(dim=-1) + self.g_cls_b, attn   # logit[B,A], attn[B,A,N]

    def _ot_branch(self, x, col_logw, g_attn=None):
        B, N, d = x.shape
        L, K = self.num_labels, self.ot_k
        kk = F.normalize(self.ot_key(x), dim=-1)                          # [B,N,d]
        sim = torch.einsum("lkd,bnd->blkn", F.normalize(self.protos, dim=-1), kk)  # [B,L,K,N]
        mass_l = None
        if self.evidence == "aot":
            # attention-GUIDED sparse OT: per-label patch demand (col marginal) from a saliency source,
            # solved with a sparse/relaxed transport; per-label total mass is a confidence gate.
            eps = self.log_eps.exp().clamp(min=1e-3, max=1.0)
            cost = (1.0 - sim).reshape(B * L, K, N)
            if self.ot_demand == "attention" and g_attn is not None:
                demand = g_attn.detach()                                  # [B,L,N] stop-grad saliency
                clw = torch.log(demand.clamp(min=1e-9)).reshape(B * L, N)
            elif self.ot_demand == "mask" and col_logw is not None:
                clw = col_logw[:, None, :].expand(B, L, N).reshape(B * L, N)
            else:                                                         # uniform
                clw = None
            if self.ot_relax == "unbalanced":
                T = sinkhorn_unbalanced(cost, eps, self.uot_rho, self.iters, clw)
            elif self.ot_relax == "partial":
                T = sinkhorn_partial(cost, eps, self.iters, self.partial_m, self.partial_tau, clw)
            else:
                T = sinkhorn_log(cost, eps, self.iters, clw)
            T = T.reshape(B, L, K, N)
            a = T.sum(dim=2)                                              # [B,L,N] patch-marginal (mass-bearing)
            mass_l = a.sum(dim=-1)                                        # [B,L] per-sign transported mass
        elif self.evidence == "ot":
            eps = self.log_eps.exp().clamp(min=1e-3, max=1.0)
            cost = (1.0 - sim).reshape(B * L, K, N)                       # per-(b,l) independent OT
            clw = (col_logw[:, None, :].expand(B, L, N).reshape(B * L, N)
                   if (self.use_mask_mass and col_logw is not None) else None)
            T = sinkhorn_log(cost, eps, self.iters, clw).reshape(B, L, K, N)  # [B,L,K,N]
            a = T.sum(dim=2)                                              # [B,L,N] patch-marginal
        else:  # attention pooling over patches (max over the K prototypes)
            score = sim.amax(dim=2) / 0.1                                 # [B,L,N]
            if col_logw is not None:
                score = score + col_logw[:, None, :]
            a = torch.softmax(score, dim=-1)                             # [B,L,N]
        ahat = a / a.sum(dim=-1, keepdim=True).clamp(min=1e-9)          # [B,L,N] evidence weights (direction)
        vv = self.ot_val(x)                                             # [B,N,d]
        ev = torch.einsum("bln,bnd->bld", ahat, vv)                    # [B,L,d]
        ev = self.ot_out_norm(ev)
        ot_logit = (ev * self.ot_cls_w).sum(dim=-1) + self.ot_cls_b    # [B,L]
        if self.evidence == "aot" and self.mass_gate and mass_l is not None:
            ot_logit = ot_logit * mass_l.clamp(min=1e-3)               # per-sign confidence gate
        if self.proto_div:
            Ph = F.normalize(self.protos, dim=-1)                       # [L,K,d]
            G = Ph @ Ph.transpose(-1, -2)                              # [L,K,K]
            off = G * (1.0 - torch.eye(K, device=G.device))
            self.proto_div_loss = (off ** 2).sum(dim=(-1, -2)).div(max(K * (K - 1), 1)).mean()
        return ot_logit, ahat                                          # ahat[B,L,N] is the heatmap

    def forward(self, patch_features: torch.Tensor, mask_weights: torch.Tensor, return_plan: bool = False):
        x = self.proj(self.in_norm(patch_features.float()))             # [B,N,d]
        col_logw = torch.log(mask_weights.float().clamp(min=1e-4))      # [B,N]

        global_logit = g_attn = ot_logit = ot_heat = None
        if self.branch in {"global", "both"}:
            global_logit, g_attn = self._global_branch(x, col_logw)
        if self.branch in {"ot", "both"}:
            ot_logit, ot_heat = self._ot_branch(x, col_logw, g_attn=g_attn)

        if self.branch == "global":
            final = global_logit
        elif self.branch == "ot":
            final = ot_logit
        else:
            final = global_logit + self.beta * ot_logit

        if return_plan:
            B = x.shape[0]; g = int(round(x.shape[1] ** 0.5))
            out = {"final": final, "global_logit": global_logit, "ot_logit": ot_logit,
                   "beta": (self.beta.detach() if self.branch == "both" else None)}
            if ot_heat is not None:
                out["ot_heat"] = ot_heat.reshape(B, self.num_labels, g, g)
            if g_attn is not None:
                out["attn_heat"] = g_attn.reshape(B, self.num_labels, g, g)
            return final, out
        return final
