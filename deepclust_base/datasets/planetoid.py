from __future__ import annotations

from pathlib import Path

import numpy as np


def convert_cora_raw_to_npz(raw_dir: str | Path, out_path: str | Path) -> dict:
    raw_dir = Path(raw_dir)
    out_path = Path(out_path)
    content_path = raw_dir / 'cora.content'
    cites_path = raw_dir / 'cora.cites'

    if not content_path.exists():
        raise FileNotFoundError(f'Missing raw content file: {content_path}')
    if not cites_path.exists():
        raise FileNotFoundError(f'Missing raw cites file: {cites_path}')

    paper_ids: list[str] = []
    features: list[list[float]] = []
    labels_raw: list[str] = []

    with content_path.open('r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            paper_ids.append(parts[0])
            features.append([float(v) for v in parts[1:-1]])
            labels_raw.append(parts[-1])

    node_index = {paper_id: idx for idx, paper_id in enumerate(paper_ids)}
    label_names = sorted(set(labels_raw))
    label_to_id = {name: idx for idx, name in enumerate(label_names)}

    x = np.asarray(features, dtype=np.float32)
    y = np.asarray([label_to_id[name] for name in labels_raw], dtype=np.int64)
    adj = np.zeros((len(paper_ids), len(paper_ids)), dtype=np.float32)

    edge_count = 0
    with cites_path.open('r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            src, dst = parts
            if src not in node_index or dst not in node_index:
                continue
            i = node_index[src]
            j = node_index[dst]
            if i == j:
                continue
            if adj[i, j] == 0.0:
                edge_count += 1
            adj[i, j] = 1.0
            adj[j, i] = 1.0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, x=x, adj=adj, y=y)

    return {
        'dataset': 'cora',
        'num_nodes': int(x.shape[0]),
        'num_features': int(x.shape[1]),
        'num_classes': int(len(label_names)),
        'num_undirected_edges': int(np.triu(adj, k=1).sum()),
        'label_names': label_names,
        'saved_to': str(out_path),
    }
