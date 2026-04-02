from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from deepclust_base.datasets import load_data_bundle, preprocess_graph_data_cached
from deepclust_base.engine import Evaluator
from deepclust_base.models import build_model
from deepclust_base.utils.io import load_config


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate model checkpoint")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    seed = int(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.get("device", "auto") != "cpu" else "cpu")
    print(f"[info] device={device}")

    dataset_cfg = cfg.get("dataset", {})
    dataset_cfg["project_root"] = str(ROOT)
    dataset_cfg.setdefault("seed", seed)

    preprocess_cfg = cfg.get("preprocess", {})
    cache_dir = ROOT / cfg.get("output", {}).get("cache_dir", "outputs/cache")
    bundle = preprocess_graph_data_cached(load_data_bundle(dataset_cfg), preprocess_cfg, cache_dir)

    x = torch.tensor(bundle.x, dtype=torch.float32, device=device)
    adj = torch.tensor(bundle.adj, dtype=torch.float32, device=device)

    model_cfg = cfg.get("model", {})
    model = build_model(model_cfg, in_dim=x.shape[1]).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    with torch.no_grad():
        z = model(x, adj)["z"].cpu().numpy()

    evaluator = Evaluator(num_clusters=int(model_cfg.get("num_clusters", 10)), random_state=seed)
    metrics = evaluator.evaluate_embeddings(z, bundle.y)
    print(metrics)


if __name__ == "__main__":
    main()
