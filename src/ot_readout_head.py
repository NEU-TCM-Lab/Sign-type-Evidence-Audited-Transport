from __future__ import annotations

# Three 8-attribute readout heads over a frozen patch-token grid, for the OT-readout gate:
#   - mlp     : mask-weighted pool -> 2-layer MLP  (= the maskpool baseline mechanism)
#   - softmax : 8 learnable attribute queries -> softmax cross-attention over patches
#   - ot      : same queries, but an entropic-OT (Sinkhorn) transport plan replaces softmax
# Optional mask_guided: bias the patch marginal/attention toward the tongue (mask) region.
# Backbone is frozen; only this head trains.

import torch
from torch import nn
import torch.nn.functional as F


def sinkhorn_log(cost: torch.Tensor, eps: float, iters: int, col_logw: torch.Tensor | None = None) -> torch.Tensor:
    # cost: [B,A,N] (lower = better match). Returns transport plan P: [B,A,N], rows ~ uniform
    # over A attributes, cols ~ (mask-biased) uniform over N patches. Log-domain stable.
    B, A, N = cost.shape
    K = -cost / eps                                   # [B,A,N]
    log_a = torch.full((B, A), -torch.log(torch.tensor(float(A))), device=cost.device)
    if col_logw is None:
        log_b = torch.full((B, N), -torch.log(torch.tensor(float(N))), device=cost.device)
    else:
        log_b = torch.log_softmax(col_logw, dim=-1)   # [B,N]
    u = torch.zeros(B, A, device=cost.device)
    v = torch.zeros(B, N, device=cost.device)
    for _ in range(iters):
        u = log_a - torch.logsumexp(K + v.unsqueeze(1), dim=2)
        v = log_b - torch.logsumexp(K + u.unsqueeze(2), dim=1)
    return torch.exp(K + u.unsqueeze(2) + v.unsqueeze(1))   # [B,A,N]


def sinkhorn_unbalanced(cost: torch.Tensor, eps: float, rho: float, iters: int,
                        col_logw: torch.Tensor | None = None) -> torch.Tensor:
    # KL-relaxed (unbalanced) OT. lam = rho/(rho+eps) damps the marginal updates:
    #   rho -> inf (lam->1)  recovers balanced OT (competition; known to LOSE here);
    #   rho -> 0   (lam->0)  removes marginal constraints (each sign ~ independent, softmax-like).
    # The interesting regime is the middle: soft column structure + ADAPTIVE per-sign mass
    # (absent/rare signs carry less total mass -> cleaner pooled feature under heavy label skew).
    # Plan is NOT row-renormalized downstream, so the adaptive mass is preserved.
    B, A, N = cost.shape
    K = -cost / eps
    log_a = torch.full((B, A), -torch.log(torch.tensor(float(A))), device=cost.device)
    log_b = (torch.log_softmax(col_logw, dim=-1) if col_logw is not None
             else torch.full((B, N), -torch.log(torch.tensor(float(N))), device=cost.device))
    lam = rho / (rho + eps)
    u = torch.zeros(B, A, device=cost.device)
    v = torch.zeros(B, N, device=cost.device)
    for _ in range(iters):
        u = lam * (log_a - torch.logsumexp(K + v.unsqueeze(1), dim=2))
        v = lam * (log_b - torch.logsumexp(K + u.unsqueeze(2), dim=1))
    return torch.exp(K + u.unsqueeze(2) + v.unsqueeze(1))   # [B,A,N]


def sinkhorn_partial(cost: torch.Tensor, eps: float, iters: int, m: float = 0.7, tau: float = 0.5,
                     col_logw: torch.Tensor | None = None) -> torch.Tensor:
    # Partial OT: transport only a fraction m in (0,1] of the total mass; the leftover (1-m) is
    # absorbed by a DUMMY reservoir column with fixed cost tau (patches whose best match is more
    # expensive than tau get discarded -> sparse, attention-localized plan). Implemented as a
    # balanced log-domain Sinkhorn over the augmented cost [B,A,N+1] (last col = tau), then drop the
    # dummy column. Returns plan: [B,A,N] with plan.sum() ~ m. Lower tau -> more mass discarded.
    B, A, N = cost.shape
    m = float(max(min(m, 1.0), 1e-3))
    cost_aug = torch.cat([cost, cost.new_full((B, A, 1), float(tau))], dim=2)   # [B,A,N+1]
    if col_logw is None:
        log_real = torch.full((B, N), -torch.log(torch.tensor(float(N))), device=cost.device)
    else:
        log_real = torch.log_softmax(col_logw, dim=-1)                          # [B,N]
    # column marginal: real cols carry m * b_real, dummy col carries (1-m)
    log_b = torch.cat([log_real + torch.log(torch.tensor(float(m), device=cost.device)),
                       torch.full((B, 1), float(torch.log(torch.tensor(1.0 - m + 1e-9))), device=cost.device)], dim=1)
    log_a = torch.full((B, A), -torch.log(torch.tensor(float(A))), device=cost.device)
    K = -cost_aug / eps
    u = torch.zeros(B, A, device=cost.device)
    v = torch.zeros(B, N + 1, device=cost.device)
    for _ in range(iters):
        u = log_a - torch.logsumexp(K + v.unsqueeze(1), dim=2)
        v = log_b - torch.logsumexp(K + u.unsqueeze(2), dim=1)
    plan_aug = torch.exp(K + u.unsqueeze(2) + v.unsqueeze(1))                    # [B,A,N+1]
    return plan_aug[:, :, :N]                                                    # drop dummy reservoir col


def sinkhorn_semirelaxed(cost: torch.Tensor, eps: float, lam_col, iters: int,
                         col_logw: torch.Tensor | None = None,
                         col_bias: torch.Tensor | None = None) -> torch.Tensor:
    # Semi-relaxed entropic OT generalizing softmax attention. The ROW (A) marginal is HARD-enforced
    # (each of the A sources gets unit mass); the COLUMN (N) marginal is KL-relaxed by lam_col in [0,1].
    # col_bias [B,N] is an ADDITIVE patch log-prior folded into the kernel (the mask-guidance bias, so
    # it is active even at lam_col=0, exactly like attention's `logits += col_logw`).
    #   lam_col = 0  -> v never updates -> plan = row-softmax(-cost/eps + col_bias) == SOFTMAX ATTENTION
    #                   (no patch competition); with eps=0.1, cost=1-cos this reproduces 73.45 EXACTLY.
    #   lam_col = 1  -> balanced OT toward the col_logw demand (full patch competition = refuted ot_only).
    # lam_col is learnable (scalar or per-batch [B]) so the model CHOOSES how much OT structure to add
    # on top of attention -> a strict generalization initialised at the attention solution.
    B, A, N = cost.shape
    K = -cost / eps
    if col_bias is not None:
        K = K + col_bias.unsqueeze(1)
    log_a = torch.full((B, A), -torch.log(torch.tensor(float(A))), device=cost.device)
    log_b = (torch.log_softmax(col_logw, dim=-1) if col_logw is not None
             else torch.full((B, N), -torch.log(torch.tensor(float(N))), device=cost.device))
    if not torch.is_tensor(lam_col):
        lam_col = torch.tensor(float(lam_col), device=cost.device)
    lam = lam_col.reshape(-1, 1) if lam_col.ndim >= 1 else lam_col          # [B,1] or scalar
    u = torch.zeros(B, A, device=cost.device)
    v = torch.zeros(B, N, device=cost.device)
    for _ in range(iters):
        u = log_a - torch.logsumexp(K + v.unsqueeze(1), dim=2)              # enforce row marginal
        v = lam * (log_b - torch.logsumexp(K + u.unsqueeze(2), dim=1))      # relaxed col marginal
    return torch.exp(K + u.unsqueeze(2) + v.unsqueeze(1))                   # [B,A,N]


class AttrReadoutHead(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        num_labels: int = 8,
        proj_dim: int = 256,
        readout: str = "ot",                # mlp | softmax | ot | mpsa | uot
        mask_guided: bool = True,
        eps_init: float = 0.05,
        sinkhorn_iters: int = 20,
        learn_eps: bool = True,
        uot_rho: float = 0.1,               # UOT marginal-relaxation strength (rho)
        ot_reg: bool = False,               # softmax + OT-alignment auxiliary regularizer (V10 attn_align)
        ot_reg_eps: float = 0.1,
        spatial_reg: bool = False,          # spatial-coherence: 2D total-variation smoothness on attention
        num_heads: int = 1,                 # multi-head attention/transport (Sinkformer-style for ot)
        query_emb: torch.Tensor | None = None,  # [A,emb_dim] MLLM text prototypes -> queries via learned proj
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if readout not in {"mlp", "softmax", "ot", "mpsa", "uot"}:
            raise ValueError(readout)
        if proj_dim % num_heads != 0:
            raise ValueError(f"proj_dim {proj_dim} not divisible by num_heads {num_heads}")
        self.readout = readout
        self.num_labels = num_labels
        self.mask_guided = mask_guided
        self.num_heads = num_heads
        self.sinkhorn_iters = sinkhorn_iters
        self.uot_rho = float(uot_rho)
        self.ot_reg = bool(ot_reg)
        self.ot_reg_eps = float(ot_reg_eps)
        self.spatial_reg = bool(spatial_reg)
        self.aux_loss = torch.tensor(0.0)
        self.spatial_loss = torch.tensor(0.0)
        self.in_norm = nn.LayerNorm(feature_dim)
        self.proj = nn.Linear(feature_dim, proj_dim)

        if readout == "mlp":
            self.head = nn.Sequential(
                nn.LayerNorm(proj_dim), nn.Linear(proj_dim, 1024), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(1024, num_labels),
            )
        else:
            self.use_text_query = query_emb is not None
            if self.use_text_query:
                self.register_buffer("query_emb", query_emb.float())          # [A,emb_dim] fixed
                self.query_proj = nn.Linear(query_emb.shape[1], proj_dim)      # learned projection
            else:
                self.queries = nn.Parameter(torch.randn(num_labels, proj_dim) * 0.02)
            self.key = nn.Linear(proj_dim, proj_dim)
            self.val = nn.Linear(proj_dim, proj_dim)
            self.cls_w = nn.Parameter(torch.randn(num_labels, proj_dim) * 0.02)
            self.cls_b = nn.Parameter(torch.zeros(num_labels))
            self.out_norm = nn.LayerNorm(proj_dim)
            self.drop = nn.Dropout(dropout)
            self.log_eps = nn.Parameter(torch.log(torch.tensor(float(eps_init))), requires_grad=learn_eps)

    def forward(self, patch_features: torch.Tensor, mask_weights: torch.Tensor) -> torch.Tensor:
        # patch_features: [B,N,Dfeat]; mask_weights: [B,N] in [0,1]
        x = self.proj(self.in_norm(patch_features.float()))          # [B,N,d]

        if self.readout == "mlp":
            w = mask_weights.float().clamp(min=0)
            denom = w.sum(dim=1, keepdim=True).clamp(min=1e-6)
            pooled = (x * w.unsqueeze(-1)).sum(dim=1) / denom         # mask-weighted mean [B,d]
            return self.head(pooled)

        col_logw = None
        if self.mask_guided:
            col_logw = torch.log(mask_weights.float().clamp(min=1e-4))   # [B,N] bias to tongue

        Q = self.query_proj(self.query_emb) if self.use_text_query else self.queries   # [A,proj_dim]

        # ---- multi-head attention/transport (num_heads>1) for softmax | ot ----
        if self.num_heads > 1 and self.readout in {"softmax", "ot"}:
            B, N, d = x.shape
            H = self.num_heads; hd = d // H
            q = F.normalize(Q.view(self.num_labels, H, hd), dim=-1)   # [A,H,hd]
            k = F.normalize(self.key(x).view(B, N, H, hd), dim=-1)              # [B,N,H,hd]
            v = self.val(x).view(B, N, H, hd)                                  # [B,N,H,hd]
            sim = torch.einsum("ahd,bnhd->bhan", q, k)                          # [B,H,A,N]
            A = self.num_labels
            if self.readout == "softmax":
                logits = sim / 0.1
                if col_logw is not None:
                    logits = logits + col_logw[:, None, None, :]
                attn = torch.softmax(logits, dim=-1)                            # [B,H,A,N]
                self.aux_loss = torch.zeros((), device=sim.device)
            else:  # ot per head
                eps = self.log_eps.exp().clamp(min=1e-3, max=1.0)
                cost = (1.0 - sim).reshape(B * H, A, N)
                clw = None if col_logw is None else col_logw[:, None, :].expand(B, H, N).reshape(B * H, N)
                plan = sinkhorn_log(cost, eps, self.sinkhorn_iters, clw).reshape(B, H, A, N)
                attn = plan / plan.sum(dim=-1, keepdim=True).clamp(min=1e-9)
            if self.spatial_reg:  # spatial TV on head-averaged attention map
                am = attn.mean(1)                                   # [B,A,N]
                g = int(round(N ** 0.5))
                if g * g == N:
                    amap = am.view(B, A, g, g)
                    self.spatial_loss = (amap[:, :, 1:, :] - amap[:, :, :-1, :]).abs().mean() \
                                      + (amap[:, :, :, 1:] - amap[:, :, :, :-1]).abs().mean()
                else:
                    self.spatial_loss = torch.zeros((), device=attn.device)
            else:
                self.spatial_loss = torch.zeros((), device=attn.device)
            pooled = torch.einsum("bhan,bnhd->bahd", attn, v).reshape(B, A, d)  # [B,A,d]
            pooled = self.drop(self.out_norm(pooled))
            return (pooled * self.cls_w).sum(dim=-1) + self.cls_b

        q = F.normalize(Q, dim=-1)                                   # [A,d]
        k = F.normalize(self.key(x), dim=-1)                         # [B,N,d]
        v = self.val(x)                                             # [B,N,d]
        sim = torch.einsum("ad,bnd->ban", q, k)                     # [B,A,N] cosine in [-1,1]

        if self.readout == "softmax":
            logits = sim / 0.1
            if col_logw is not None:
                logits = logits + col_logw.unsqueeze(1)
            attn = torch.softmax(logits, dim=-1)                    # [B,A,N]
            if self.ot_reg:  # V10 attn_align: pull attention toward a (smoothing) OT plan
                with torch.no_grad():
                    P = sinkhorn_log(1.0 - sim, self.ot_reg_eps, self.sinkhorn_iters, col_logw)
                    Pn = (P / P.sum(dim=-1, keepdim=True).clamp(min=1e-9)).clamp(min=1e-9)
                self.aux_loss = (attn.clamp(min=1e-9) * (attn.clamp(min=1e-9).log() - Pn.log())).sum(-1).mean()
            else:
                self.aux_loss = torch.zeros((), device=sim.device)
        elif self.readout == "ot":
            eps = self.log_eps.exp().clamp(min=1e-3, max=1.0)
            plan = sinkhorn_log(1.0 - sim, eps, self.sinkhorn_iters, col_logw)  # [B,A,N]
            attn = plan / plan.sum(dim=-1, keepdim=True).clamp(min=1e-9)        # row-normalize
        elif self.readout == "uot":  # unbalanced OT: soft marginals + ADAPTIVE per-sign mass.
            eps = self.log_eps.exp().clamp(min=1e-3, max=1.0)
            plan = sinkhorn_unbalanced(1.0 - sim, eps, self.uot_rho, self.sinkhorn_iters, col_logw)
            attn = plan * float(self.num_labels)                    # keep adaptive mass (no row-renorm)
        else:  # mpsa: pure inter-attribute competition (UNIFORM column marginal), plan used
               # directly (x A so rows ~sum 1) -> patches are split AMONG the 8 attributes.
            eps = self.log_eps.exp().clamp(min=1e-3, max=1.0)
            plan = sinkhorn_log(1.0 - sim, eps, self.sinkhorn_iters, col_logw=None)  # uniform col
            attn = plan * float(self.num_labels)                    # [B,A,N], preserves competition

        if self.spatial_reg:  # 2D total-variation on the per-sign attention map (spatial coherence)
            B2, A2, N2 = attn.shape
            g = int(round(N2 ** 0.5))
            if g * g == N2:
                amap = attn.view(B2, A2, g, g)
                tv = (amap[:, :, 1:, :] - amap[:, :, :-1, :]).abs().mean() \
                   + (amap[:, :, :, 1:] - amap[:, :, :, :-1]).abs().mean()
                self.spatial_loss = tv
            else:
                self.spatial_loss = torch.zeros((), device=attn.device)
        else:
            self.spatial_loss = torch.zeros((), device=attn.device)

        pooled = torch.einsum("ban,bnd->bad", attn, v)              # [B,A,d]
        pooled = self.drop(self.out_norm(pooled))
        return (pooled * self.cls_w).sum(dim=-1) + self.cls_b       # [B,A] = logits
