from __future__ import annotations

from .daegc import DAEGC
from .dec import DEC
from .idec import IDEC
from .gae_kmeans import GraphAEBaseline


def build_model(model_cfg: dict, in_dim: int, num_nodes: int = None):
    name = model_cfg.get("name", "gae_kmeans_baseline")
    if name == "gae_kmeans_baseline":
        return GraphAEBaseline(
            in_dim=in_dim,
            hidden_dim=int(model_cfg.get("hidden_dim", 256)),
            embed_dim=int(model_cfg.get("embed_dim", 64)),
            num_clusters=int(model_cfg.get("num_clusters", 10)),
            dropout=float(model_cfg.get("dropout", 0.0)),
        )
    if name == "daegc":
        return DAEGC(
            num_features=in_dim,
            hidden_size=int(model_cfg.get("hidden_dim", 256)),
            embedding_size=int(model_cfg.get("embed_dim", 64)),
            num_clusters=int(model_cfg.get("num_clusters", 10)),
            alpha=float(model_cfg.get("alpha", 0.2)),
            v=float(model_cfg.get("v", 1.0)),
            pretrain_path=model_cfg.get("pretrain_path"),
        )

    if name == "dec":
        return DEC(
            input_dim=in_dim,
            hidden_dims=model_cfg.get("hidden_dims", [500, 500, 2000]),
            embedding_dim=int(model_cfg.get("embed_dim", 10)),
            num_clusters=int(model_cfg.get("num_clusters", 10)),
            dropout=float(model_cfg.get("dropout", 0.2)),
            pretrain_path=model_cfg.get("pretrain_path"),
        )

    if name == "idec":
        return IDEC(
            input_dim=in_dim,
            hidden_dims=model_cfg.get("hidden_dims", [500, 500, 2000]),
            embedding_dim=int(model_cfg.get("embed_dim", 10)),
            num_clusters=int(model_cfg.get("num_clusters", 10)),
            gamma=float(model_cfg.get("gamma", 0.1)),
            pretrain_path=model_cfg.get("pretrain_path"),
        )

    raise ValueError(f"Unsupported model: {name}")
