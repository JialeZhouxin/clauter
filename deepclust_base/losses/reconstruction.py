from __future__ import annotations

import torch
import torch.nn.functional as F


def feature_reconstruction_loss(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(x_hat, x)


def adjacency_reconstruction_loss(adj_logits: torch.Tensor, adj_target: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(adj_logits, adj_target)
