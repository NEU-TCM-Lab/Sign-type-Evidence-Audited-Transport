from __future__ import annotations

import torch
from torch import nn


class MaskPoolBBoxHead(nn.Module):
    def __init__(
        self,
        feature_dim: int = 2560,
        bbox_dim: int = 9,
        bbox_hidden: int = 256,
        hidden_dim: int = 1024,
        num_labels: int = 8,
        dropout: float = 0.2,
        feature_mode: str = "global_mask_bbox",
    ) -> None:
        super().__init__()
        if feature_mode not in {"global", "global_mask", "global_mask_bbox"}:
            raise ValueError(f"Unsupported feature_mode: {feature_mode}")
        self.feature_dim = feature_dim
        self.bbox_dim = bbox_dim
        self.bbox_hidden = bbox_hidden
        self.num_labels = num_labels
        self.feature_mode = feature_mode
        self.global_norm = nn.LayerNorm(feature_dim)
        in_dim = feature_dim
        if feature_mode in {"global_mask", "global_mask_bbox"}:
            self.mask_norm = nn.LayerNorm(feature_dim)
            in_dim += feature_dim
        else:
            self.mask_norm = None
        if feature_mode == "global_mask_bbox":
            self.bbox_mlp = nn.Sequential(
                nn.LayerNorm(bbox_dim),
                nn.Linear(bbox_dim, bbox_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            in_dim += bbox_hidden
        else:
            self.bbox_mlp = None
        self.head = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels),
        )

    def forward(self, global_features: torch.Tensor, mask_features: torch.Tensor, bbox_norm: torch.Tensor) -> torch.Tensor:
        parts = [self.global_norm(global_features.float())]
        if self.feature_mode in {"global_mask", "global_mask_bbox"}:
            if self.mask_norm is None:
                raise RuntimeError("mask_norm was not initialized")
            parts.append(self.mask_norm(mask_features.float()))
        if self.feature_mode == "global_mask_bbox":
            if self.bbox_mlp is None:
                raise RuntimeError("bbox_mlp was not initialized")
            parts.append(self.bbox_mlp(bbox_norm.float()))
        return self.head(torch.cat(parts, dim=-1))
