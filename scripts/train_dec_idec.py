"""
DEC / IDEC 统一训练脚本

完全按论文实现三阶段训练流程：

Phase 1: 逐层贪婪预训练 (layer-wise pretrain)
  - 每层单独作为去噪 AE 训练 50000 iterations (论文原文)
  - 逐层向上: 原始数据 → 压缩表示 → ... → embedding
  - MSE 重构损失, dropout=0.2

Phase 2: 端到端精调 (end-to-end pretrain)
  - 组装完整 SAE/BP
  - 精调 100000 iterations (论文原文)
  - 无 dropout, lr=0.1 → 0.01 (每 20000 衰减 10x)
  - 保存 pretrain checkpoint

Phase 3: 聚类微调 (clustering finetune)
  - 从预训练 encoder 提取嵌入
  - KMeans 初始化聚类中心 (20 restarts)
  - DEC:  丢弃 decoder, 纯 KL(Q||P) 损失, lr=0.01
  - IDEC: 保留 decoder, L = L_r + γ*L_c, lr=0.001 (Adam) 或 lr=0.1 (SGD)
  - 每 T 次迭代更新目标分布 P
  - 收敛: 标签变化 < tol = 0.1%

用法:
  python train_dec_idec.py --config configs/dec_mnist.yaml
  python train_dec_idec.py --config configs/idec_mnist.yaml
"""

from __future__ import annotations

import argparse
import sys
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepclust_base.models.dec import DEC, compute_target_distribution as dec_target_dist
from deepclust_base.models.idec import IDEC, compute_target_distribution as idec_target_dist
from deepclust_base.utils.metrics import clustering_accuracy
from deepclust_base.utils.logger import write_history_csv
from deepclust_base.utils.io import load_config, save_json
from deepclust_base.utils.checkpoint import save_checkpoint
from deepclust_base.utils.seed import set_global_seed


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------

def compute_acc_nmi_ari(y_true: np.ndarray, y_pred: np.ndarray):
    """用 Hungarian algorithm 计算最佳映射下的 ACC, NMI, ARI"""
    from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
    acc = clustering_accuracy(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)
    return {"acc": float(acc), "nmi": float(nmi), "ari": float(ari)}


# ---------------------------------------------------------------------------
# Phase 1: 逐层贪婪预训练
# ---------------------------------------------------------------------------

def phase1_layerwise_pretrain(model, x, cfg):
    """论文 §3.2 / IDEC §3.1: 逐层贪婪预训练

    修改对照论文:
    - 论文原文: "Each layer is pretrained for 50000 iterations"
    - 原实现: 基于 epochs (50 epochs ≈ 13650 iterations, 严重不足)
    - 修改后: 基于 iterations (50000 iterations per layer)
    """
    set_global_seed(42)
    print("\n=== Phase 1: Layer-wise Pretrain ===")

    # 论文原文: 50000 iterations per layer
    max_iters = int(cfg.get("layer_pretrain_iters", 50000))
    batch_size = int(cfg.get("layer_pretrain_batch_size", 256))
    lr = float(cfg.get("layer_pretrain_lr", 0.1))
    corruption = float(cfg.get("dropout", 0.2))  # 即噪声比例

    all_dims = [model.input_dim] + model.hidden_dims + [model.embedding_dim]

    # x 是 (N, input_dim)，逐层作为伪输入
    h = x.detach().clone()
    layer_inputs = [x]

    for layer_idx in range(len(all_dims) - 1):
        in_d = all_dims[layer_idx]
        out_d = all_dims[layer_idx + 1]

        # 临时 AE
        ae_enc = nn.Linear(in_d, out_d, bias=True)
        ae_dec = nn.Linear(out_d, in_d, bias=True)
        # 论文原文: "weights initialized from zero-mean Gaussian, std=0.01"
        nn.init.normal_(ae_enc.weight, mean=0.0, std=0.01)
        nn.init.normal_(ae_dec.weight, mean=0.0, std=0.01)
        nn.init.zeros_(ae_enc.bias)
        nn.init.zeros_(ae_dec.bias)
        ae_enc = ae_enc.to(x.device)
        ae_dec = ae_dec.to(x.device)

        opt = torch.optim.SGD(list(ae_enc.parameters()) + list(ae_dec.parameters()),
                              lr=lr, momentum=0.9)
        cur_input = layer_inputs[-1].detach()
        N = cur_input.shape[0]

        # 论文原文: batch_size=256
        n_batches = max(1, N // batch_size)

        for it in range(1, max_iters + 1):
            # 随机 batch
            perm = torch.randperm(N)
            b_start = ((it - 1) * batch_size) % N
            b_idx = perm[b_start: b_start + batch_size]
            xb = cur_input[b_idx]

            # 去噪: 随机置零 corruption 比例
            mask = torch.rand_like(xb) > corruption
            x_corrupt = xb * mask.float()

            # Encode (除最后一层都用 ReLU)
            z = ae_enc(x_corrupt)
            if layer_idx < len(all_dims) - 2:
                z = F.relu(z)

            # Decode
            x_recon = ae_dec(z)
            if layer_idx > 0:  # decoder 第一层不激活
                x_recon = F.relu(x_recon)

            loss = F.mse_loss(x_recon, xb.detach())

            opt.zero_grad()
            loss.backward()
            opt.step()

            if it % 5000 == 0 or it == max_iters:
                print(f"  Layer {layer_idx+1}/{len(all_dims)-1} "
                      f"Iter {it}/{max_iters}: recon_loss={loss.item():.6f}")

        # 把权重迁移到模型
        _copy_layer(model, layer_idx, ae_enc.weight, ae_enc.bias)
        _copy_decoder(model, layer_idx, ae_dec.weight, ae_dec.bias)

        # 计算下一层输入
        with torch.no_grad():
            z_next = ae_enc(cur_input)
            if layer_idx < len(all_dims) - 2:
                z_next = F.relu(z_next)
            layer_inputs.append(z_next)

    print("  Phase 1 done.")


def _copy_layer(model, layer_idx, weight, bias):
    if hasattr(model, 'sae'):
        enc = model.sae.encoder
    else:
        enc = model.ae.encoder

    linear_idx = layer_idx * 2 if layer_idx > 0 else 0
    linear = enc[linear_idx]
    with torch.no_grad():
        linear.weight.copy_(weight)
        linear.bias.copy_(bias)


def _copy_decoder(model, layer_idx, weight, bias):
    if hasattr(model, 'sae'):
        dec = model.sae.decoder
    else:
        dec = model.ae.decoder

    dec_len = len(dec)
    # decoder: Linear, ReLU 交替; 最后一组 Linear 后无 ReLU
    dec_linear_idx = dec_len - 1 - layer_idx
    dec_linear_idx = max(0, min(dec_linear_idx, dec_len - 1))
    if hasattr(dec[dec_linear_idx], 'weight'):
        linear = dec[dec_linear_idx]
        with torch.no_grad():
            linear.weight.copy_(weight)
            linear.bias.copy_(bias)


# ---------------------------------------------------------------------------
# Phase 2: 端到端精调
# ---------------------------------------------------------------------------

def phase2_end2end_pretrain(model, x, cfg, output_dir):
    """论文 §3.2 / IDEC §3.1: 端到端精调 SAE

    修改对照论文:
    - 论文原文: "finetuned for 100000 iterations, lr=0.1, divided by 10 every 20000 iterations, SGD+momentum=0.9"
    - 原实现: Adam, lr=0.01 固定, 20 epochs (严重偏离论文)
    - 修改后: SGD+momentum=0.9, lr=0.1, 每 20000 iterations 衰减 10x, 共 100000 iterations
    """
    print("\n=== Phase 2: End-to-End Pretrain ===")

    max_iters = int(cfg.get("ae_finetune_iters", 100000))
    init_lr = float(cfg.get("ae_finetune_lr", 0.1))
    momentum = float(cfg.get("ae_finetune_momentum", 0.9))
    decay_iters = int(cfg.get("ae_finetune_decay_iters", 20000))  # 每 20000 衰减 10x
    wd = float(cfg.get("weight_decay", 0.0))
    batch_size = int(cfg.get("ae_finetune_batch_size", 256))

    device = x.device
    N = x.shape[0]

    if hasattr(model, 'sae'):
        params = list(model.sae.parameters())
    else:
        params = list(model.ae.parameters())

    optimizer = torch.optim.SGD(params, lr=init_lr, momentum=momentum, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=decay_iters,
                                                  gamma=0.1)  # 衰减 10x

    history = []
    n_batches = max(1, N // batch_size)

    for it in range(1, max_iters + 1):
        idx = torch.randperm(N)
        total_loss = 0.0

        # 一个 iteration 遍历所有 batch
        for b in range(n_batches):
            batch_idx = idx[b * batch_size: (b + 1) * batch_size]
            xb = x[batch_idx]

            z = model.encode(xb)
            x_recon = model.decode(z)
            loss = F.mse_loss(x_recon, xb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Step scheduler 每个 iteration 结束后调用
        scheduler.step()

        avg_loss = total_loss / n_batches
        current_lr = optimizer.param_groups[0]['lr']

        if it % 10000 == 0 or it == max_iters:
            print(f"  Iter {it}/{max_iters}: recon_loss={avg_loss:.6f}, lr={current_lr:.6f}")

        if it % 1000 == 0:
            history.append({"iter": it, "loss_rec": avg_loss, "lr": current_lr, "stage": "ae_finetune"})

    # 保存预训练权重
    pretrain_path = output_dir / "pretrain_ae.pkl"
    if hasattr(model, 'sae'):
        torch.save(model.sae.state_dict(), pretrain_path)
    else:
        torch.save(model.ae.state_dict(), pretrain_path)
    print(f"  Saved pretrain AE to {pretrain_path}")
    return str(pretrain_path)


# ---------------------------------------------------------------------------
# Phase 3: 聚类微调
# ---------------------------------------------------------------------------

def phase3_clustering_finetune(model, x, y, cfg, output_dir):
    """DEC / IDEC 聚类微调"""
    print("\n=== Phase 3: Clustering Finetune ===")

    is_idec = model.__class__.__name__ == "IDEC"

    # ---- 超参数 ----
    max_epochs = int(cfg.get("clustering_epochs", 100))
    gamma = float(cfg.get("gamma", 0.1))        # 仅 IDEC
    lr = float(cfg.get("clustering_lr", 0.001))
    target_update_interval = int(cfg.get("target_update_interval", 1))  # T
    stop_tol = float(cfg.get("stop_tol", 0.001))  # tol=0.1%
    eval_every = int(cfg.get("eval_every", 1))

    device = x.device
    N = x.shape[0]

    # ---- 用预训练编码器提取嵌入，KMeans 初始化中心 ----
    model.eval()
    with torch.no_grad():
        z_init = model.encode(x).detach().cpu().numpy()

    seed = int(cfg.get("seed", 42))
    km = KMeans(n_clusters=model.num_clusters, n_init=20, random_state=seed)
    y_pred_init = km.fit_predict(z_init)

    # 初始化聚类中心
    if hasattr(model, 'cluster_layer'):
        centers = torch.tensor(km.cluster_centers_, dtype=torch.float32, device=device)
        with torch.no_grad():
            model.cluster_layer.centers.copy_(centers)
    print(f"  Initialized {model.num_clusters} cluster centers from KMeans (ACC={clustering_accuracy(y, y_pred_init):.4f})")

    # ---- 优化器设置 ----
    model.set_stage("clustering")
    model.train()

    # ---- 聚类优化器选择 ----
    clustering_optimizer_type = cfg.get("clustering_optimizer", "adam").lower()
    clustering_momentum = float(cfg.get("clustering_momentum", 0.9))

    if is_idec:
        params = list(model.ae.parameters()) + list(model.cluster_layer.parameters())
    else:
        if hasattr(model, 'sae'):
            params = list(model.sae.encoder.parameters()) + list(model.cluster_layer.parameters())
        else:
            params = list(model.ae.encoder.parameters()) + list(model.cluster_layer.parameters())

    if clustering_optimizer_type == "sgd":
        optimizer = torch.optim.SGD(params, lr=lr, momentum=clustering_momentum)
    else:
        optimizer = torch.optim.Adam(params, lr=lr)

    # ---- 初始化目标分布 P ----
    with torch.no_grad():
        q = model.get_Q(model.encode(x))
    if is_idec:
        p = idec_target_dist(q)
    else:
        p = dec_target_dist(q)

    history = []
    best_metrics = {"acc": -1.0, "nmi": -1.0, "ari": -1.0}
    best_epoch = 0
    global_iter = 0

    for epoch in range(1, max_epochs + 1):
        global_iter += 1

        # ---- 更新目标分布 P (每 T 次迭代) ----
        if global_iter % target_update_interval == 0:
            model.eval()
            with torch.no_grad():
                q = model.get_Q(model.encode(x))
            if is_idec:
                p = idec_target_dist(q)
            else:
                p = dec_target_dist(q)

            # 计算当前标签 + 检查收敛
            y_pred = q.detach().cpu().numpy().argmax(axis=1)
            n_change = np.sum(y_pred != y_pred_init) / N
            y_pred_init = y_pred.copy()

            # 评估
            metrics = compute_acc_nmi_ari(y, y_pred)
            print(f"  [P-update @{epoch}] acc={metrics['acc']:.4f} "
                  f"nmi={metrics['nmi']:.4f} ari={metrics['ari']:.4f} "
                  f"label_change={n_change:.4f}")

            row = {"epoch": epoch, "stage": "clustering"}
            row.update(metrics)
            row["label_change"] = n_change
            history.append(row)

            if metrics["acc"] > best_metrics["acc"]:
                best_metrics = metrics
                best_epoch = epoch

            # 收敛检查
            if global_iter > target_update_interval and n_change < stop_tol:
                print(f"  Converged: label_change={n_change:.4f} < tol={stop_tol}")
                break

        # ---- 梯度更新 ----
        optimizer.zero_grad()

        z = model.encode(x)
        q = model.get_Q(z)

        if is_idec:
            x_recon = model.ae.decode(z)
            loss_rec = F.mse_loss(x_recon, x)
            loss_kl = F.kl_div(q.log(), p, reduction="batchmean")
            loss_total = loss_rec + gamma * loss_kl
            losses = {"loss_total": loss_total, "loss_rec": loss_rec, "loss_kl": loss_kl}
        else:
            loss_kl = F.kl_div(q.log(), p, reduction="batchmean")
            losses = {"loss_total": loss_kl, "loss_kl": loss_kl}

        loss_total = losses["loss_total"]
        loss_total.backward()
        optimizer.step()

        # [P1 修复] IDEC 显式聚类中心更新 - 论文公式(10)-(13)
        # 论文原文: "the cluster centers are updated by explicit gradient descent:
        #           μ_j = μ_j - lr * ∂L_c/∂μ_j"
        # 原实现: 仅用 autograd backward，update_cluster_centers 从未被调用
        # [P0 修复] IDEC 显式聚类中心更新 - 论文公式(10)-(13)
        # 论文原文: "the cluster centers are updated by explicit gradient descent:
        #           μ_j = μ_j - lr * ∂L_c/∂μ_j"
        if is_idec and hasattr(model, 'update_cluster_centers'):
            # 使用当前 batch 的 z, q, p 显式更新聚类中心（论文公式 10）
            model.update_cluster_centers(z, q, p, lr=lr)

        # 记录
        if global_iter % eval_every == 0:
            row_history = {"epoch": global_iter, "stage": "clustering"}
            for k, v in losses.items():
                row_history[k] = float(v.detach().cpu().item())
            history.append(row_history)

    # ---- 最终评估 ----
    model.eval()
    with torch.no_grad():
        z_final = model.encode(x)
        q_final = model.get_Q(z_final)
        y_pred_final = q_final.cpu().numpy().argmax(axis=1)

    final_metrics = compute_acc_nmi_ari(y, y_pred_final)
    print(f"\n  Final metrics: acc={final_metrics['acc']:.4f} "
          f"nmi={final_metrics['nmi']:.4f} ari={final_metrics['ari']:.4f}")
    print(f"  Best metrics (epoch {best_epoch}): acc={best_metrics['acc']:.4f}")

    return history, best_metrics, best_epoch, final_metrics


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def run(cfg: dict, root: Path):
    set_global_seed(int(cfg.get("seed", 42)))
    device = torch.device("cpu")
    print(f"[INFO] device={device}")

    # ---- 数据 ----
    dataset_cfg = cfg.get("dataset", {})
    dataset_cfg["project_root"] = str(root)

    # DEC/IDEC 不需要图结构，从 datasets loader 获取纯特征数据
    from deepclust_base.datasets.loader import load_data_bundle
    bundle = load_data_bundle(dataset_cfg)

    # 数据预处理: L2 归一化 (论文要求: 1/d * ||xi||_2^2 ≈ 1)
    x_raw = bundle.x.astype(np.float32)
    norms = np.linalg.norm(x_raw, axis=1, keepdims=True)
    x_raw = x_raw / (norms + 1e-12)
    # 缩放到单位范数后乘 sqrt(d)，让 1/d*||xi||^2 ≈ 1
    d = x_raw.shape[1]
    x_raw = x_raw * np.sqrt(d)

    x = torch.tensor(x_raw, dtype=torch.float32, device=device)
    y = bundle.y

    print(f"[DATA] N={x.shape[0]}, d={x.shape[1]}, K={cfg['model']['num_clusters']}")

    # ---- 构建模型 ----
    model_cfg = cfg.get("model", {})
    num_clusters = int(model_cfg.get("num_clusters", 10))

    model_name = model_cfg.get("name", "dec")
    if model_name == "dec":
        model = DEC(
            input_dim=x.shape[1],
            hidden_dims=model_cfg.get("hidden_dims", [500, 500, 2000]),
            embedding_dim=int(model_cfg.get("embed_dim", 10)),
            num_clusters=num_clusters,
            dropout=float(model_cfg.get("dropout", 0.2)),
        )
    elif model_name == "idec":
        model = IDEC(
            input_dim=x.shape[1],
            hidden_dims=model_cfg.get("hidden_dims", [500, 500, 2000]),
            embedding_dim=int(model_cfg.get("embed_dim", 10)),
            num_clusters=num_clusters,
            gamma=float(model_cfg.get("gamma", 0.1)),
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model = model.to(device)
    print(f"[MODEL] {model.__class__.__name__}")

    # ---- 输出目录 ----
    output_cfg = cfg.get("output", {})
    output_dir = root / output_cfg.get("dir", "outputs/dec_run")
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Phase 1: 逐层预训练 ----
    phase1_layerwise_pretrain(model, x, cfg)

    # ---- Phase 2: 端到端精调 ----
    pretrain_path = phase2_end2end_pretrain(model, x, cfg, output_dir)

    # ---- Phase 3: 聚类微调 ----
    model.load_pretrain(pretrain_path)
    history, best_metrics, best_epoch, final_metrics = phase3_clustering_finetune(
        model, x, y, cfg, output_dir
    )

    # ---- 保存结果 ----
    save_json(best_metrics, output_dir / "best_metrics.json")
    save_json({"best_epoch": best_epoch, "final": final_metrics}, output_dir / "final_metrics.json")
    write_history_csv(history, output_dir / "history.csv")

    ckpt = output_dir / "last.ckpt"
    if hasattr(model, 'sae'):
        torch.save(model.sae.state_dict(), ckpt)
    else:
        torch.save(model.ae.state_dict(), ckpt)

    print(f"\n[DONE] Results saved to {output_dir}")
    return best_metrics


def main():
    parser = argparse.ArgumentParser(description="DEC / IDEC Training")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    args = parser.parse_args()

    cfg = load_config(args.config)
    root = Path(__file__).resolve().parents[1]
    run(cfg, root)


if __name__ == "__main__":
    main()
