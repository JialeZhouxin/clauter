from __future__ import annotations

import torch


def soft_kmeans_like_loss(q: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    # Encourage confident and balanced assignments.
    p = q.mean(dim=0)
    entropy_per_node = -(q * (q + eps).log()).sum(dim=1).mean()
    entropy_global = -(p * (p + eps).log()).sum()
    return entropy_per_node - entropy_global
