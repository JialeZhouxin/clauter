from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_model import BaseClusteringModel
from ..losses.clustering import soft_kmeans_like_loss
from ..losses.reconstruction import feature_reconstruction_loss, adjacency_reconstruction_loss


class GraphAEBaseline(BaseClusteringModel):
    """A simple AE+GCN style baseline for framework validation and early experiments."""

    def __init__(self, in_dim: int, hidden_dim: int, embed_dim: int, num_clusters: int, dropout: float = 0.0):
        super().__init__()
        self.num_clusters = num_clusters
        self.dropout = dropout

        self.enc_1 = nn.Linear(in_dim, hidden_dim)
        self.enc_2 = nn.Linear(hidden_dim, embed_dim)
        self.dec = nn.Linear(embed_dim, in_dim)
        self.cluster_head = nn.Linear(embed_dim, num_clusters)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = torch.matmul(adj, x)
        h = F.relu(self.enc_1(h))
        h = F.dropout(h, p=self.dropout, training=self.training)
        z = self.enc_2(torch.matmul(adj, h))

        x_hat = self.dec(z)
        q_logits = self.cluster_head(z)
        q = F.softmax(q_logits, dim=1)

        adj_logits = torch.matmul(z, z.t())

        return {
            "z": z,
            "x_hat": x_hat,
            "q": q,
            "adj_logits": adj_logits,
        }

    def compute_losses(self, outputs: Dict[str, torch.Tensor], x: torch.Tensor, adj: torch.Tensor, cfg: dict) -> Dict[str, torch.Tensor]:
        w_rec_x = float(cfg.get("w_rec_x", 1.0))
        w_rec_adj = float(cfg.get("w_rec_adj", 1.0))
        w_cluster = float(cfg.get("w_cluster", 0.1))

        l_x = feature_reconstruction_loss(outputs["x_hat"], x)
        l_adj = adjacency_reconstruction_loss(outputs["adj_logits"], adj)
        l_cluster = soft_kmeans_like_loss(outputs["q"])

        total = w_rec_x * l_x + w_rec_adj * l_adj + w_cluster * l_cluster
        return {
            "loss_total": total,
            "loss_rec_x": l_x,
            "loss_rec_adj": l_adj,
            "loss_cluster": l_cluster,
        }
