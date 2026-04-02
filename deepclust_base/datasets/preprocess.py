from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .loader import DataBundle


def _normalize_features(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    row_sum = np.maximum(x.sum(axis=1, keepdims=True), eps)
    return x / row_sum


def _normalize_adjacency(adj: np.ndarray, add_self_loops: bool = True, method: str = "l1") -> np.ndarray:
    """归一化邻接矩阵
    
    Args:
        adj: 邻接矩阵
        add_self_loops: 是否加自环
        method: 归一化方法
            - "symmetric": 对称归一化 D^(-1/2) * A * D^(-1/2)
            - "l1": 行归一化 (论文中使用的方法)
    """
    a = adj.copy()
    if add_self_loops:
        np.fill_diagonal(a, 1.0)
    
    if method == "symmetric":
        # 对称归一化
        deg = a.sum(axis=1)
        deg_inv_sqrt = np.power(np.maximum(deg, 1e-12), -0.5)
        d_mat = np.diag(deg_inv_sqrt)
        return d_mat @ a @ d_mat
    elif method == "l1":
        # 行归一化 (论文中使用的方法)
        row_sum = np.maximum(a.sum(axis=1, keepdims=True), 1e-12)
        return a / row_sum
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def preprocess_graph_data(bundle: DataBundle, cfg: dict) -> DataBundle:
    x = bundle.x.astype(np.float32)
    adj = bundle.adj.astype(np.float32)

    if cfg.get("normalize_features", False):
        x = _normalize_features(x)
    if cfg.get("normalize_adjacency", True):
        adj = _normalize_adjacency(
            adj,
            add_self_loops=cfg.get("add_self_loops", True),
            method=cfg.get("adj_normalize_method", "l1"),  # 默认使用论文的 l1 归一化
        )

    return DataBundle(x=x, adj=adj, y=bundle.y, name=bundle.name)


def _cache_key(bundle: DataBundle, cfg: dict) -> str:
    payload = {
        "dataset_name": bundle.name,
        "shape_x": list(bundle.x.shape),
        "shape_adj": list(bundle.adj.shape),
        "cfg": {
            "normalize_features": bool(cfg.get("normalize_features", True)),
            "normalize_adjacency": bool(cfg.get("normalize_adjacency", True)),
            "add_self_loops": bool(cfg.get("add_self_loops", True)),
        },
    }
    s = json.dumps(payload, sort_keys=True)
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:12]


def preprocess_graph_data_cached(bundle: DataBundle, cfg: dict, cache_dir: str | Path) -> DataBundle:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    use_cache = bool(cfg.get("use_cache", True))
    key = _cache_key(bundle, cfg)
    cache_path = cache_dir / f"{bundle.name}_{key}.npz"

    if use_cache and cache_path.exists():
        data = np.load(cache_path, allow_pickle=False)
        x = np.asarray(data["x"], dtype=np.float32)
        adj = np.asarray(data["adj"], dtype=np.float32)
        y = np.asarray(data["y"], dtype=np.int64) if "y" in data else bundle.y
        return DataBundle(x=x, adj=adj, y=y, name=bundle.name)

    processed = preprocess_graph_data(bundle, cfg)

    if use_cache:
        payload = {"x": processed.x, "adj": processed.adj}
        if processed.y is not None:
            payload["y"] = processed.y
        np.savez_compressed(cache_path, **payload)

    return processed
