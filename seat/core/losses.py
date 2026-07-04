from __future__ import annotations

# Stage-1 DINOv2 discriminative baseline losses:
#   - AsymmetricLoss (ASL) for multi-label classification (Ben-Baruch et al., ICCV'21).
#   - Logit adjustment for long-tail multi-label (Menon et al., ICLR'21), train-time variant.
#   - build_loss(): config-driven factory selecting bce / asl / logit_adjusted_bce / asl_logit_adjusted.
#
# These are additive and self-contained: importing this module does not change any existing
# training behaviour. train_head.py opts in via the --loss / --logit-adjust flags.

import torch
from torch import nn


class AsymmetricLoss(nn.Module):
    """Multi-label Asymmetric Loss operating on raw logits.

    positive term:  (1 - p) ** gamma_pos * log(p)
    negative term:  p_m ** gamma_neg   * log(1 - p_m),  with p_m = max(p - clip, 0)

    clip down-weights easy negatives; gamma_neg > gamma_pos focuses on hard positives by
    decaying the contribution of easy negatives. Numerically stable via eps clamping.
    """

    def __init__(
        self,
        gamma_pos: float = 0.0,
        gamma_neg: float = 4.0,
        clip: float = 0.05,
        eps: float = 1e-8,
        reduction: str = "mean",
        pos_weight: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError(f"Unsupported reduction: {reduction}")
        self.gamma_pos = float(gamma_pos)
        self.gamma_neg = float(gamma_neg)
        self.clip = float(clip)
        self.eps = float(eps)
        self.reduction = reduction
        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight)
        else:
            self.pos_weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.shape != targets.shape:
            raise AssertionError(f"ASL logits/targets shape mismatch: {tuple(logits.shape)} vs {tuple(targets.shape)}")
        targets = targets.float()
        p = torch.sigmoid(logits)

        # positive branch
        log_p = torch.log(p.clamp(min=self.eps))
        loss_pos = (1.0 - p).pow(self.gamma_pos) * log_p

        # negative branch with probability shifting (asymmetric clipping)
        p_m = (p - self.clip).clamp(min=0.0)
        log_1m = torch.log((1.0 - p_m).clamp(min=self.eps))
        loss_neg = p_m.pow(self.gamma_neg) * log_1m

        per_pos = targets * loss_pos
        if self.pos_weight is not None:
            per_pos = per_pos * self.pos_weight.to(per_pos.device)
        loss = -(per_pos + (1.0 - targets) * loss_neg)

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def compute_pos_prior(labels: torch.Tensor, clamp_min: float = 1e-3, clamp_max: float = 1.0 - 1e-3) -> torch.Tensor:
    """Per-class positive prior pi_c = num_positive_c / num_samples, clamped away from 0/1."""
    labels = labels.float()
    pi = labels.mean(dim=0)
    return pi.clamp(min=clamp_min, max=clamp_max)


def logit_adjustment_vector(pi: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
    """adjustment_c = tau * log((1 - pi_c) / pi_c); subtract from logits before the loss."""
    pi = pi.clamp(min=1e-6, max=1.0 - 1e-6)
    return tau * torch.log((1.0 - pi) / pi)


class LogitAdjuster(nn.Module):
    """Holds a per-class logit-adjustment vector as a registered buffer (moves with .to(device)).

    adjusted_logits = logits - adjustment_c.  Used at train time so the loss is computed on
    adjusted logits; optionally also applied at eval (configurable).
    """

    def __init__(self, pi: torch.Tensor, tau: float = 1.0) -> None:
        super().__init__()
        self.tau = float(tau)
        self.register_buffer("adjustment", logit_adjustment_vector(pi, tau))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits - self.adjustment.to(logits.device)


def build_loss(
    name: str,
    *,
    pos_weight: torch.Tensor | None = None,
    asl_gamma_pos: float = 0.0,
    asl_gamma_neg: float = 4.0,
    asl_clip: float = 0.05,
    reduction: str = "mean",
) -> nn.Module:
    """Config-driven base loss factory. Logit adjustment is applied OUTSIDE this (LogitAdjuster
    transforms the logits first), so 'bce' and 'logit_adjusted_bce' share the same criterion here.

    name in {bce, asl, logit_adjusted_bce, asl_logit_adjusted}.
    """
    name = name.lower()
    if name in {"bce", "logit_adjusted_bce"}:
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction=reduction)
    if name in {"asl", "asl_logit_adjusted"}:
        return AsymmetricLoss(
            gamma_pos=asl_gamma_pos,
            gamma_neg=asl_gamma_neg,
            clip=asl_clip,
            reduction=reduction,
            pos_weight=pos_weight,
        )
    raise ValueError(f"Unknown loss '{name}'. Expected one of: bce, asl, logit_adjusted_bce, asl_logit_adjusted")


def loss_uses_logit_adjustment(name: str) -> bool:
    return name.lower() in {"logit_adjusted_bce", "asl_logit_adjusted"}
