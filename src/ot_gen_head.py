from __future__ import annotations

# OTGenHead: an OT-GENERALIZED attention readout over a frozen DINOv2 patch grid.
#
# Unlike the refuted additive SignOTHead (final = global + beta*OT, where OT just copied attention),
# here OT IS the readout. Each of the 8 signs has K learnable prototypes; per-sign, a SEMI-RELAXED
# entropic OT (row=prototypes hard-enforced, col=patches KL-relaxed by a learnable lam_col) transports
# the prototypes onto the patches. Because softmax-attention == semi-relaxed OT with lam_col=0, this is
# a STRICT GENERALIZATION of the 73.45 attention readout: lam_col=0 recovers it exactly (per prototype),
# lam_col>0 adds patch-competition that the optimizer turns on only if it helps. K=1 + lam=0 == baseline.
#
# Grounded in ROTP (pooling = regularized OT, attention is a special case) and OTKE (aggregate to a
# trainable reference set via an OT plan). The transport plan is the interpretable evidence map.

import torch
from torch import nn
import torch.nn.functional as F

from ot_readout_head import sinkhorn_semirelaxed


class OTGenHead(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        num_labels: int = 8,
        proj_dim: int = 256,
        ot_k: int = 4,                   # prototypes per sign (K=1 reproduces the attention baseline)
        eps_init: float = 0.1,           # = the baseline softmax temperature (cost=1-cos)
        sinkhorn_iters: int = 20,
        learn_eps: bool = True,
        lam_init: float = 0.02,          # initial patch-competition (small -> start near attention)
        per_class_lam: bool = True,      # each sign chooses its own patch-competition strength
        combine: str = "mean",           # mean | concat  (OTKE-style concat of per-prototype vectors)
        peak_topk: int = 0,              # >0: add a per-class top-k peak-similarity DETECTION term
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.readout = "otgen"
        self.num_labels = num_labels
        self.ot_k = int(ot_k)
        self.iters = int(sinkhorn_iters)
        # loss hooks expected by the train loop
        self.aux_loss = torch.tensor(0.0)
        self.spatial_loss = torch.tensor(0.0)
        self.proto_div_loss = torch.tensor(0.0)

        self.in_norm = nn.LayerNorm(feature_dim)
        self.proj = nn.Linear(feature_dim, proj_dim)
        self.drop = nn.Dropout(dropout)
        self.log_eps = nn.Parameter(torch.log(torch.tensor(float(eps_init))), requires_grad=learn_eps)

        # lam_col in (0,1) via sigmoid; init logit so sigmoid(.)~lam_init (start near attention).
        import math
        lo = math.log(lam_init / (1.0 - lam_init))
        nlam = num_labels if per_class_lam else 1
        self.lam_logit = nn.Parameter(torch.full((nlam,), float(lo)))
        self.per_class_lam = per_class_lam

        self.combine = combine
        cls_dim = proj_dim * (self.ot_k if combine == "concat" else 1)    # OTKE concat -> [L, K*d]
        self.protos = nn.Parameter(torch.randn(num_labels, self.ot_k, proj_dim) * 0.02)
        self.key = nn.Linear(proj_dim, proj_dim)
        self.val = nn.Linear(proj_dim, proj_dim)
        self.cls_w = nn.Parameter(torch.randn(num_labels, cls_dim) * 0.02)
        self.cls_b = nn.Parameter(torch.zeros(num_labels))
        self.out_norm = nn.LayerNorm(cls_dim)
        # peak detection term: top-k prototype->patch similarity inside the tongue mask. Localized signs
        # (Ecchymosis spots) produce a strong single-patch match that mean-pooling dilutes.
        self.peak_topk = int(peak_topk)
        if self.peak_topk > 0:
            self.peak_gain = nn.Parameter(torch.full((num_labels,), 0.5))
            self.peak_bias = nn.Parameter(torch.zeros(num_labels))

    def forward(self, patch_features: torch.Tensor, mask_weights: torch.Tensor, return_plan: bool = False):
        x = self.proj(self.in_norm(patch_features.float()))                # [B,N,d]
        B, N, d = x.shape
        L, K = self.num_labels, self.ot_k
        col_logw = torch.log(mask_weights.float().clamp(min=1e-4))         # [B,N] patch demand (mask)

        kk = F.normalize(self.key(x), dim=-1)                             # [B,N,d]
        sim = torch.einsum("lkd,bnd->blkn", F.normalize(self.protos, dim=-1), kk)  # [B,L,K,N]
        eps = self.log_eps.exp().clamp(min=1e-3, max=1.0)
        cost = (1.0 - sim).reshape(B * L, K, N)                            # per-(b,l) independent OT
        clw = col_logw[:, None, :].expand(B, L, N).reshape(B * L, N)      # relaxed col-marginal target (mask)
        cbias = clw                                                       # additive patch prior in kernel:
        # lam=0 -> softmax(sim/eps + col_logw) == the mask-guided attention readout (73.45) EXACTLY.
        lam = torch.sigmoid(self.lam_logit)                               # [L] or [1] in (0,1)
        lam_bl = (lam[None, :].expand(B, L).reshape(B * L) if self.per_class_lam
                  else lam.expand(B * L))
        T = sinkhorn_semirelaxed(cost, eps, lam_bl, self.iters, clw, col_bias=cbias).reshape(B, L, K, N)
        attn = T / T.sum(dim=-1, keepdim=True).clamp(min=1e-9)            # [B,L,K,N] each proto's patch dist
        vv = self.val(x)                                                  # [B,N,d]
        pooled = torch.einsum("blkn,bnd->blkd", attn, vv)                 # [B,L,K,d] per-prototype pooled
        ev = (pooled.reshape(B, L, K * pooled.shape[-1]) if self.combine == "concat"
              else pooled.mean(dim=2))                                    # concat -> [B,L,K*d]; mean -> [B,L,d]
        ev = self.drop(self.out_norm(ev))
        logit = (ev * self.cls_w).sum(dim=-1) + self.cls_b               # [B,L]

        if self.peak_topk > 0:
            simmax = sim.amax(dim=2)                                      # [B,L,N] best prototype per patch
            simmax = simmax + col_logw[:, None, :]                        # restrict peaks to the tongue mask
            k = min(self.peak_topk, N)
            peak = simmax.topk(k, dim=-1).values.mean(dim=-1)            # [B,L] mean of top-k matched patches
            logit = logit + self.peak_gain * peak + self.peak_bias       # additive per-class detection term

        if return_plan:
            g = int(round(N ** 0.5))
            heat = attn.mean(dim=2).reshape(B, L, g, g)                   # [B,L,g,g] evidence heatmap
            return logit, {"final": logit, "ot_heat": heat, "attn_heat": heat,
                           "lam": lam.detach(), "eps": eps.detach()}
        return logit


class DualOTHead(nn.Module):
    # Dual-pathway OT readout: DINOv2 (semantic) and COLOR get SEPARATE OT-peak readouts with their own
    # normalization/prototypes, fused per-class: logit = dino_logit + gamma_c * color_logit. Avoids the
    # concat dilution (12 color dims drowned in 1024) so the color pathway gets a full voice for the
    # color-defined rare signs. Input patch_features = [DINOv2(dino_dim) | color] concatenated by --extra-tag.
    def __init__(self, feature_dim: int, dino_dim: int = 1024, num_labels: int = 8, proj_dim: int = 256,
                 ot_k: int = 8, eps_init: float = 0.1, sinkhorn_iters: int = 20, learn_eps: bool = True,
                 lam_init: float = 0.02, per_class_lam: bool = True, combine: str = "concat",
                 peak_topk: int = 16, dropout: float = 0.2) -> None:
        super().__init__()
        self.readout = "dualot"
        self.dino_dim = int(dino_dim)
        self.color_dim = int(feature_dim) - int(dino_dim)
        self.aux_loss = torch.tensor(0.0); self.spatial_loss = torch.tensor(0.0)
        self.proto_div_loss = torch.tensor(0.0)
        mk = lambda fd: OTGenHead(fd, num_labels, proj_dim=proj_dim, ot_k=ot_k, eps_init=eps_init,
                                  sinkhorn_iters=sinkhorn_iters, learn_eps=learn_eps, lam_init=lam_init,
                                  per_class_lam=per_class_lam, combine=combine, peak_topk=peak_topk, dropout=dropout)
        self.dino = mk(self.dino_dim)
        self.color = mk(self.color_dim)
        self.gamma = nn.Parameter(torch.full((num_labels,), 0.5))         # per-class color contribution
        self.log_eps = self.dino.log_eps                                  # for summary/eps reporting

    def forward(self, patch_features, mask_weights, return_plan: bool = False):
        xd = patch_features[..., :self.dino_dim]
        xc = patch_features[..., self.dino_dim:]
        ld = self.dino(xd, mask_weights)
        lc = self.color(xc, mask_weights)
        logit = ld + self.gamma * lc
        if return_plan:
            _, info = self.color(xc, mask_weights, return_plan=True)
            return logit, {"final": logit, "dino_logit": ld.detach(), "color_logit": lc.detach(),
                           "gamma": self.gamma.detach(), **{k: info[k] for k in ("ot_heat", "attn_heat") if k in info}}
        return logit
