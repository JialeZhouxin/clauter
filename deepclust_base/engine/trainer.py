from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch

from .evaluator import Evaluator


@dataclass
class TrainOutput:
    history: List[Dict[str, float]]
    best_metrics: Dict[str, float]
    best_epoch: int


class Trainer:
    def __init__(self, model, optimizer, evaluator: Evaluator, cfg: dict, device: torch.device):
        self.model = model
        self.optimizer = optimizer
        self.evaluator = evaluator
        self.cfg = cfg
        self.device = device

    def _eval_with_embeddings(self, x: torch.Tensor, adj: torch.Tensor, y):
        with torch.no_grad():
            z = self.model(x, adj)["z"].detach().cpu().numpy()
        return self.evaluator.evaluate_embeddings(z, y)

    def _eval_with_clusters(self, x: torch.Tensor, adj: torch.Tensor, y):
        with torch.no_grad():
            outputs = self.model(x, adj)
            y_pred = outputs["q"].detach().cpu().numpy().argmax(axis=1)
        return self.evaluator.evaluate_labels(y_pred, y)

    def train(self, x: torch.Tensor, adj: torch.Tensor, y):
        pretrain_epochs = int(self.cfg.get("pretrain_epochs", self.cfg.get("epochs", 200)))
        finetune_epochs = int(self.cfg.get("finetune_epochs", 0))
        eval_every = int(self.cfg.get("eval_every", 10))
        w_kl = float(self.cfg.get("w_kl", 10.0))

        history = []
        best_metrics = {"nmi": -1.0, "ari": -1.0, "acc": -1.0}
        best_epoch = -1

        # 阶段 1：预训练
        if hasattr(self.model, "set_stage"):
            self.model.set_stage("pretrain")

        for epoch in range(1, pretrain_epochs + 1):
            self.optimizer.zero_grad()
            outputs = self.model(x, adj)
            losses = self.model.compute_losses(outputs, x, adj, {})
            losses["loss_total"].backward()
            self.optimizer.step()

            row = {k: float(v.detach().cpu().item()) for k, v in losses.items()}
            row["stage"] = "pretrain"
            row["epoch"] = epoch

            if epoch % eval_every == 0 or epoch == pretrain_epochs:
                self.model.eval()
                metrics = self._eval_with_embeddings(x, adj, y)
                row.update(metrics)
                if metrics.get("nmi", -1.0) > best_metrics["nmi"]:
                    best_metrics = metrics
                    best_epoch = epoch
                self.model.train()

            history.append(row)

        # 用预训练嵌入初始化聚类中心
        if hasattr(self.model, "init_cluster_layer_from_embeddings"):
            self.model.eval()
            with torch.no_grad():
                z = self.model.encode(x, adj)
            self.model.init_cluster_layer_from_embeddings(z, seed=getattr(self.evaluator, "random_state", 42))

        # 阶段 2：微调
        if finetune_epochs > 0 and hasattr(self.model, "set_stage"):
            self.model.set_stage("finetune")
            for epoch in range(1, finetune_epochs + 1):
                self.optimizer.zero_grad()
                outputs = self.model(x, adj)
                losses = self.model.compute_losses(outputs, x, adj, {"w_kl": w_kl})
                losses["loss_total"].backward()
                self.optimizer.step()

                row = {k: float(v.detach().cpu().item()) for k, v in losses.items()}
                row["stage"] = "finetune"
                row["epoch"] = epoch

                if epoch % eval_every == 0 or epoch == finetune_epochs:
                    self.model.eval()
                    metrics = self._eval_with_clusters(x, adj, y)
                    row.update(metrics)
                    if metrics.get("nmi", -1.0) > best_metrics["nmi"]:
                        best_metrics = metrics
                        best_epoch = pretrain_epochs + epoch
                    self.model.train()

                history.append(row)

        return TrainOutput(history=history, best_metrics=best_metrics, best_epoch=best_epoch)
