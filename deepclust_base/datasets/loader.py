from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class DataBundle:
    x: np.ndarray
    adj: np.ndarray
    y: Optional[np.ndarray] = None
    name: str = "unknown"


def _load_npz(path: Path, dataset_name: str) -> DataBundle:
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    data = np.load(path, allow_pickle=True)
    required = ["x", "adj"]
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(f"NPZ file missing required keys: {missing}")

    x = np.asarray(data["x"], dtype=np.float32)
    adj = np.asarray(data["adj"], dtype=np.float32)
    y = np.asarray(data["y"], dtype=np.int64) if "y" in data else None
    return DataBundle(x=x, adj=adj, y=y, name=dataset_name)


def _make_synthetic(num_nodes: int = 600, num_features: int = 64, num_clusters: int = 6, seed: int = 42) -> DataBundle:
    rng = np.random.default_rng(seed)

    centers = rng.normal(0, 2.0, size=(num_clusters, num_features)).astype(np.float32)
    labels = rng.integers(0, num_clusters, size=(num_nodes,))
    x = centers[labels] + 0.25 * rng.normal(size=(num_nodes, num_features)).astype(np.float32)

    adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    p_intra = 0.08
    p_inter = 0.008
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            p = p_intra if labels[i] == labels[j] else p_inter
            if rng.random() < p:
                adj[i, j] = 1.0
                adj[j, i] = 1.0

    return DataBundle(x=x, adj=adj, y=labels.astype(np.int64), name="synthetic")


def load_data_bundle(dataset_cfg: dict) -> DataBundle:
    source = dataset_cfg.get("source", "synthetic")
    name = dataset_cfg.get("name", source)

    if source == "synthetic":
        return _make_synthetic(
            num_nodes=int(dataset_cfg.get("num_nodes", 600)),
            num_features=int(dataset_cfg.get("num_features", 64)),
            num_clusters=int(dataset_cfg.get("num_clusters", 6)),
            seed=int(dataset_cfg.get("seed", 42)),
        )

    if source == "npz":
        path = Path(dataset_cfg.get("path", ""))
        if not path.is_absolute():
            path = Path(dataset_cfg.get("project_root", ".")) / path
        return _load_npz(path, dataset_name=name)

    # DEC/IDEC 专用: 加载 .mat 格式数据集（MNIST, USPS 等）
    if source == "mat":
        from scipy.io import loadmat
        path = Path(dataset_cfg.get("path", ""))
        if not path.is_absolute():
            path = Path(dataset_cfg.get("project_root", ".")) / path
        mat = loadmat(str(path))
        x = mat["data"].astype(np.float32)
        y = mat["label"].ravel().astype(np.int64) if "label" in mat else None
        # DEC/IDEC 数据集通常 data 和 label 在同一矩阵或子键中
        # 常见键名: 'X', 'y', 'data', 'label', 'fea', 'gnd'
        for key in ["X", "fea", "data"]:
            if key in mat:
                x = mat[key].astype(np.float32)
                break
        for key in ["y", "label", "gnd"]:
            if key in mat:
                y = mat[key].ravel().astype(np.int64)
                break
        # 无图结构，返回单位矩阵作为占位
        N = x.shape[0]
        adj = np.eye(N, dtype=np.float32)
        return DataBundle(x=x, adj=adj, y=y, name=name)

    raise ValueError(f"Unsupported dataset source: {source}. Use 'synthetic', 'npz', or 'mat'.")
