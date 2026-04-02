from __future__ import annotations

import argparse
import sys
from pathlib import Path

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
from sklearn.cluster import KMeans
import numpy as np
import torch.nn.functional as F


def run_daegc_two_phase(cfg: dict, root: Path):
    seed = int(cfg.get("seed", 42))
    device = torch.device("cpu")
    print(f"[info] device={device}")

    dataset_cfg = cfg.get("dataset", {})
    dataset_cfg["project_root"] = str(root)
    dataset_cfg.setdefault("seed", seed)

    bundle = load_data_bundle(dataset_cfg)
    preprocess_cfg = cfg.get("preprocess", {})
    cache_dir = root / cfg.get("output", {}).get("cache_dir", "outputs/cache")
    bundle = preprocess_graph_data_cached(bundle, preprocess_cfg, cache_dir)

    x = torch.tensor(bundle.x, dtype=torch.float32, device=device)
    adj = torch.tensor(bundle.adj, dtype=torch.float32, device=device)
    y = bundle.y

    model_cfg = cfg.get("model", {})
    model = build_model(model_cfg, in_dim=x.shape[1]).to(device)
    model.set_stage("pretrain")

    opt_cfg = cfg.get("optimizer", {})
    pretrain_lr = float(opt_cfg.get("pretrain_lr", 5e-3))
    finetune_lr = float(opt_cfg.get("finetune_lr", 1e-4))
    weight_decay = float(opt_cfg.get("weight_decay", 5e-3))

    trainer_cfg = cfg.get("trainer", {})
    pretrain_epochs = int(trainer_cfg.get("pretrain_epochs", 30))
    finetune_epochs = int(trainer_cfg.get("finetune_epochs", 100))
    eval_every = int(trainer_cfg.get("eval_every", 10))
    w_kl = float(trainer_cfg.get("w_kl", 10.0))
    target_update_interval = int(trainer_cfg.get("target_update_interval", 1))
    finetune_eval_mode = str(trainer_cfg.get("finetune_eval_mode", "cluster"))
    finetune_stop_tol = float(trainer_cfg.get("finetune_stop_tol", 0.0))

    history = []
    best_metrics = {"nmi": -1.0, "ari": -1.0, "acc": -1.0}
    best_epoch = -1

    # Stage 1: pretrain
    optimizer = torch.optim.Adam(model.parameters(), lr=pretrain_lr, weight_decay=weight_decay)
    for epoch in range(1, pretrain_epochs + 1):
        model.train()
        optimizer.zero_grad()
        outputs = model(x, adj)
        loss = F.binary_cross_entropy(outputs["A_pred"].view(-1), adj.view(-1))
        loss.backward()
        optimizer.step()

        row = {"stage": "pretrain", "epoch": epoch, "loss_total": float(loss.detach().cpu().item()), "loss_rec": float(loss.detach().cpu().item())}
        if epoch % eval_every == 0 or epoch == pretrain_epochs:
            model.eval()
            with torch.no_grad():
                z = model.encode(x, adj).detach().cpu().numpy()
            metrics = Evaluator(num_clusters=int(model_cfg.get("num_clusters", 10)), random_state=seed).evaluate_embeddings(z, y)
            row.update(metrics)
            if metrics.get("nmi", -1.0) > best_metrics["nmi"]:
                best_metrics = metrics
                best_epoch = epoch
                output_dir = root / cfg.get("output", {}).get("dir", "outputs/default")
                output_dir.mkdir(parents=True, exist_ok=True)
                torch.save(model.gat.state_dict(), output_dir / "best_model.pkl")
        history.append(row)

    # Initialize cluster centers
    model.eval()
    with torch.no_grad():
        z = model.encode(x, adj)
    kmeans = KMeans(n_clusters=int(model_cfg.get("num_clusters", 10)), n_init=20, random_state=seed)
    init_pred = kmeans.fit_predict(z.detach().cpu().numpy())
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32, device=device)

    # Stage 2: finetune
    model.set_stage("finetune")
    optimizer = torch.optim.Adam(model.parameters(), lr=finetune_lr, weight_decay=weight_decay)
    prev_metric = None
    current_p = None
    for epoch in range(1, finetune_epochs + 1):
        model.train()
        optimizer.zero_grad()
        outputs = model(x, adj)
        q = outputs["q"]

        if current_p is None or ((epoch - 1) % target_update_interval == 0):
            weight = q ** 2 / torch.clamp(q.sum(dim=0, keepdim=True), min=1e-12)
            current_p = weight / torch.clamp(weight.sum(dim=1, keepdim=True), min=1e-12)

        kl_loss = F.kl_div(q.log(), current_p, reduction="batchmean")
        rec_loss = F.binary_cross_entropy(outputs["A_pred"].view(-1), adj.view(-1))
        loss = w_kl * kl_loss + rec_loss
        loss.backward()
        optimizer.step()

        row = {"stage": "finetune", "epoch": epoch, "loss_total": float(loss.detach().cpu().item()), "loss_kl": float(kl_loss.detach().cpu().item()), "loss_rec": float(rec_loss.detach().cpu().item())}
        if epoch % eval_every == 0 or epoch == finetune_epochs:
            model.eval()
            with torch.no_grad():
                if finetune_eval_mode == "embedding":
                    z_eval = model.encode(x, adj).detach().cpu().numpy()
                    metrics = Evaluator(num_clusters=int(model_cfg.get("num_clusters", 10)), random_state=seed).evaluate_embeddings(z_eval, y)
                else:
                    q_eval = model(x, adj)["q"].detach().cpu().numpy()
                    metrics = Evaluator(num_clusters=int(model_cfg.get("num_clusters", 10)), random_state=seed).evaluate_labels(q_eval.argmax(axis=1), y)
            row.update(metrics)
            if metrics.get("nmi", -1.0) > best_metrics["nmi"]:
                best_metrics = metrics
                best_epoch = pretrain_epochs + epoch
                output_dir = root / cfg.get("output", {}).get("dir", "outputs/default")
                output_dir.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), output_dir / "best_model.pkl")
            if prev_metric is not None and abs(metrics.get("nmi", 0.0) - prev_metric) < finetune_stop_tol:
                history.append(row)
                break
            prev_metric = metrics.get("nmi", 0.0)
        history.append(row)

    output_dir = root / cfg.get("output", {}).get("dir", "outputs/default")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_history_csv(output_dir / "history.csv", history)
    save_json(output_dir / "best_metrics.json", {**best_metrics, "best_epoch": best_epoch})
    save_checkpoint(output_dir / "last.ckpt", model=model, optimizer=optimizer, epoch=pretrain_epochs + finetune_epochs, extra={"config": cfg, "init_pred": init_pred.tolist()})
    print(f"[done] output_dir={output_dir}")
    print(f"[done] best_metrics={best_metrics}, best_epoch={best_epoch}")
    return best_metrics


def parse_args():
    p = argparse.ArgumentParser(description="Train unified deep clustering baseline")
    p.add_argument("--config", type=str, required=True, help="Path to yaml/json config")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    model_name = cfg.get("model", {}).get("name", "")
    if model_name == "daegc":
        run_daegc_two_phase(cfg, ROOT)
        return

    seed = int(cfg.get("seed", 42))
    set_global_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() and cfg.get("device", "auto") != "cpu" else "cpu")
    print(f"[info] device={device}")

    dataset_cfg = cfg.get("dataset", {})
    dataset_cfg["project_root"] = str(ROOT)
    dataset_cfg.setdefault("seed", seed)

    bundle = load_data_bundle(dataset_cfg)
    preprocess_cfg = cfg.get("preprocess", {})
    cache_dir = ROOT / cfg.get("output", {}).get("cache_dir", "outputs/cache")
    bundle = preprocess_graph_data_cached(bundle, preprocess_cfg, cache_dir)

    x = torch.tensor(bundle.x, dtype=torch.float32, device=device)
    adj = torch.tensor(bundle.adj, dtype=torch.float32, device=device)
    y = bundle.y

    model_cfg = cfg.get("model", {})
    model = build_model(model_cfg, in_dim=x.shape[1]).to(device)

    optim_cfg = cfg.get("optimizer", {})
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(optim_cfg.get("lr", 1e-3)),
        weight_decay=float(optim_cfg.get("weight_decay", 0.0)),
    )

    evaluator = Evaluator(num_clusters=int(model_cfg.get("num_clusters", 10)), random_state=seed)
    trainer = Trainer(model=model, optimizer=optimizer, evaluator=evaluator, cfg=cfg.get("trainer", {}), device=device)
    out = trainer.train(x, adj, y)

    output_dir = ROOT / cfg.get("output", {}).get("dir", "outputs/default")
    output_dir.mkdir(parents=True, exist_ok=True)

    write_history_csv(output_dir / "history.csv", out.history)
    save_json(output_dir / "best_metrics.json", {**out.best_metrics, "best_epoch": out.best_epoch})
    save_checkpoint(output_dir / "last.ckpt", model=model, optimizer=optimizer, epoch=cfg.get("trainer", {}).get("epochs", 0), extra={"config": cfg})

    print(f"[done] output_dir={output_dir}")
    print(f"[done] best_metrics={out.best_metrics}, best_epoch={out.best_epoch}")


if __name__ == "__main__":
    main()
