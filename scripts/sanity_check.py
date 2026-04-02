from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from deepclust_base.datasets import load_data_bundle, preprocess_graph_data_cached
from deepclust_base.engine import Evaluator, Trainer
from deepclust_base.models import build_model
from deepclust_base.utils.checkpoint import save_checkpoint
from deepclust_base.utils.io import load_config, save_json
from deepclust_base.utils.logger import write_history_csv
from deepclust_base.utils.seed import set_global_seed


def main(config_path: str = "configs/gae_synthetic.yaml"):
    cfg = load_config(str(ROOT / config_path))
    seed = int(cfg.get("seed", 42))
    set_global_seed(seed)

    # 1) Data loading + preprocess + cache
    dataset_cfg = cfg.get("dataset", {})
    dataset_cfg["project_root"] = str(ROOT)
    dataset_cfg.setdefault("seed", seed)
    cache_dir = ROOT / cfg.get("output", {}).get("cache_dir", "outputs/cache")

    raw_bundle = load_data_bundle(dataset_cfg)
    bundle = preprocess_graph_data_cached(raw_bundle, cfg.get("preprocess", {}), cache_dir)

    # 2) Evaluator with random embeddings
    evaluator = Evaluator(num_clusters=int(cfg["model"]["num_clusters"]), random_state=seed)
    random_z = np.random.randn(bundle.x.shape[0], int(cfg["model"]["embed_dim"])).astype(np.float32)
    random_metrics = evaluator.evaluate_embeddings(random_z, bundle.y)

    # 3) Trainer + toy model
    device = torch.device("cpu")
    print(f"[info] device={device}")
    x = torch.tensor(bundle.x, dtype=torch.float32, device=device)
    adj = torch.tensor(bundle.adj, dtype=torch.float32, device=device)

    model = build_model(cfg["model"], in_dim=x.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["optimizer"]["lr"]), weight_decay=float(cfg["optimizer"].get("weight_decay", 0.0)))

    trainer_cfg = dict(cfg.get("trainer", {}))
    trainer_cfg["epochs"] = min(20, int(trainer_cfg.get("epochs", 80)))
    trainer_cfg["eval_every"] = 5

    trainer = Trainer(model=model, optimizer=optimizer, evaluator=evaluator, cfg=trainer_cfg, device=device)
    out = trainer.train(x, adj, bundle.y)

    # 4) logging + checkpoint
    output_dir = ROOT / "outputs/sanity_check"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_history_csv(output_dir / "history.csv", out.history)
    save_json(output_dir / "random_metrics.json", random_metrics)
    save_json(output_dir / "best_metrics.json", {**out.best_metrics, "best_epoch": out.best_epoch})
    save_checkpoint(output_dir / "last.ckpt", model=model, optimizer=optimizer, epoch=trainer_cfg["epochs"], extra={"type": "sanity"})

    # 5) one-command run confirmation
    print("[sanity] data/preprocess/cache: OK")
    print(f"[sanity] random evaluator metrics: {random_metrics}")
    print(f"[sanity] trainer best metrics: {out.best_metrics}, best_epoch={out.best_epoch}")
    print(f"[sanity] outputs saved at: {output_dir}")


if __name__ == "__main__":
    main()
