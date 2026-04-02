"""
DAEGC 微调脚本
直接使用本地已有的 Cora 数据文件，与 GitHub 官方实现一致
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score as ari_score,
    normalized_mutual_info_score as nmi_score,
)
from sklearn.preprocessing import normalize

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from deepclust_base.models.daegc import DAEGC, target_distribution
from deepclust_base.utils.seed import set_global_seed


def load_cora_from_files(data_dir: str) -> tuple:
    """从本地文件加载 Cora 数据 (与 GitHub 一致)"""
    data_dir = Path(data_dir)
    
    # 读取节点特征
    features = []
    labels = []
    node_ids = []
    
    with open(data_dir / "cora.content", "r") as f:
        for line in f:
            parts = line.strip().split()
            node_id = int(parts[0])
            feature = list(map(float, parts[1:-1]))
            label = parts[-1]
            
            node_ids.append(node_id)
            features.append(feature)
            labels.append(label)
    
    # 创建节点 ID 到索引的映射
    node_id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
    
    num_nodes = len(node_ids)
    num_features = len(features[0])
    
    # 创建特征矩阵
    x = np.array(features, dtype=np.float32)
    
    # 读取边
    edges = []
    with open(data_dir / "cora.cites", "r") as f:
        for line in f:
            parts = line.strip().split()
            src = int(parts[0])
            dst = int(parts[1])
            if src in node_id_to_idx and dst in node_id_to_idx:
                edges.append((node_id_to_idx[src], node_id_to_idx[dst]))
    
    # 构建邻接矩阵 (与 GitHub 一致)
    adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for src, dst in edges:
        adj[src, dst] = 1.0
        adj[dst, src] = 1.0
    
    # 复制作为标签 (归一化后的邻接矩阵，与 GitHub 一致)
    adj_label_np = adj.copy()
    
    # 加自环 + 行归一化 (与 GitHub 一致)
    adj = adj + np.eye(num_nodes)
    adj = normalize(adj, norm="l1")
    adj = torch.from_numpy(adj).to(dtype=torch.float)
    
    # adj_label 也是归一化后的 (与 GitHub 一致)
    adj_label = adj.clone()
    
    # 计算 M 矩阵
    M = get_M(adj)
    
    # 标签编码
    label_to_idx = {l: i for i, l in enumerate(set(labels))}
    y = np.array([label_to_idx[l] for l in labels], dtype=np.int64)
    
    x = torch.from_numpy(x).to(dtype=torch.float)
    y = torch.from_numpy(y).to(dtype=torch.int64)
    
    return x, adj, adj_label, y, M


def get_M(adj: torch.Tensor, t: int = 2) -> torch.Tensor:
    """计算转移概率矩阵 M (与 GitHub 一致)"""
    adj_np = adj.cpu().numpy()
    tran_prob = normalize(adj_np, norm="l1", axis=0)
    M_np = sum([np.linalg.matrix_power(tran_prob, i) for i in range(1, t + 1)]) / t
    return torch.from_numpy(M_np).to(dtype=torch.float32)


def cluster_acc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """计算聚类准确率 - 完全匹配 GitHub 实现 (使用 Munkres 库)"""
    from munkres import Munkres
    
    y_true = y_true - np.min(y_true)
    
    l1 = list(set(y_true))
    numclass1 = len(l1)
    
    l2 = list(set(y_pred))
    numclass2 = len(l2)
    
    # 类别数不同时的对齐逻辑 (与 GitHub 一致)
    ind = 0
    if numclass1 != numclass2:
        for i in l1:
            if i in l2:
                pass
            else:
                y_pred[ind] = i
                ind += 1
    
    l2 = list(set(y_pred))
    numclass2 = len(l2)
    
    if numclass1 != numclass2:
        print("error")
        return 0.0
    
    # 构建成本矩阵
    cost = np.zeros((numclass1, numclass2), dtype=int)
    for i, c1 in enumerate(l1):
        mps = [i1 for i1, e1 in enumerate(y_true) if e1 == c1]
        for j, c2 in enumerate(l2):
            mps_d = [i1 for i1 in mps if y_pred[i1] == c2]
            cost[i][j] = len(mps_d)
    
    # 使用 Munkres 算法 (与 GitHub 一致)
    m = Munkres()
    cost_matrix = cost.__neg__().tolist()
    indexes = m.compute(cost_matrix)
    
    # 获取匹配结果
    new_predict = np.zeros(len(y_pred))
    for i, c in enumerate(l1):
        c2 = l2[indexes[i][1]]
        ai = [ind for ind, elm in enumerate(y_pred) if elm == c2]
        new_predict[ai] = c
    
    acc = np.mean(y_true == new_predict)
    return acc


def finetune(dataset_name: str, args):
    """微调函数"""
    seed = int(args.seed)
    set_global_seed(seed)
    
    device = torch.device(args.device)
    print(f"[info] device={device}")
    
    # 加载数据 (从本地文件)
    print(f"[info] Loading {dataset_name} from local files...")
    x, adj, adj_label, y, M = load_cora_from_files(args.data_dir)
    x = x.to(device)
    adj = adj.to(device)
    adj_label = adj_label.to(device)
    M = M.to(device)
    y = y.numpy()
    
    print(f"[info] Dataset: {dataset_name}, nodes={x.shape[0]}, features={x.shape[1]}")
    
    # 构建模型
    model = DAEGC(
        num_features=x.shape[1],
        hidden_size=args.hidden_size,
        embedding_size=args.embedding_size,
        num_clusters=args.n_clusters,
        alpha=args.alpha,
        v=args.v,
        pretrain_path=args.pretrain_path,
    ).to(device)
    
    print(f"[info] Model:\n{model}")
    
    # 优化器 (与 GitHub 一致: lr=0.0001)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    
    # 用预训练模型的嵌入初始化聚类中心
    print("[info] Initializing cluster centers with pretrained embeddings...")
    model.eval()
    with torch.no_grad():
        _, z = model.gat(x, adj, M)
        z_np = z.cpu().numpy()
    
    kmeans = KMeans(n_clusters=args.n_clusters, n_init=20, random_state=seed)
    y_pred_init = kmeans.fit_predict(z_np)
    
    # 用 KMeans 中心初始化 cluster_layer
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)
    
    acc_init = cluster_acc(y, y_pred_init)
    nmi_init = nmi_score(y, y_pred_init)
    ari_init = ari_score(y, y_pred_init)
    print(f"[info] Init: acc={acc_init:.4f}, nmi={nmi_init:.4f}, ari={ari_init:.4f}")
    
    # 训练
    best_acc = 0.0
    best_nmi = 0.0
    best_ari = 0.0
    best_epoch = 0
    
    for epoch in range(args.epochs):
        model.train()
        
        # 前向传播
        outputs = model(x, adj, M)
        A_pred = outputs["A_pred"]
        Q = outputs["q"]
        
        # 计算目标分布 P (与 GitHub 一致)
        P = target_distribution(Q.detach())
        
        # 损失: w_kl * KL + BCE (与 GitHub 一致: w_kl=10)
        # GitHub 调用两次 model: 第一次获取 Q 计算 p，第二次获取新的 Q 计算 kl
        # 但实际上两次 Q 应该一样（模型参数没变）
        kl_loss = F.kl_div(Q.log(), P, reduction='batchmean')
        rec_loss = F.binary_cross_entropy(A_pred.view(-1), adj_label.view(-1))
        loss = args.w_kl * kl_loss + rec_loss
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 评估
        if (epoch + 1) % args.eval_every == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                outputs = model(x, adj, M)
                Q_eval = outputs["q"]
            
            y_pred = Q_eval.detach().cpu().numpy().argmax(axis=1)
            
            acc = cluster_acc(y, y_pred)
            nmi = nmi_score(y, y_pred, average_method="arithmetic")
            ari = ari_score(y, y_pred)
            
            print(f"[epoch {epoch+1:3d}] loss={loss.item():.4f}, kl={kl_loss.item():.4f}, rec={rec_loss.item():.4f}")
            print(f"         -> acc={acc:.4f}, nmi={nmi:.4f}, ari={ari:.4f}")
            
            # 保存最佳模型
            if acc > best_acc:
                best_acc = acc
                best_nmi = nmi
                best_ari = ari
                best_epoch = epoch + 1
                output_dir = Path(args.output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                best_model_path = output_dir / "best_model.pkl"
                torch.save(model.state_dict(), best_model_path)
                print(f"         -> saved best model (acc={best_acc:.4f})")
    
    print(f"\n[done] Best: acc={best_acc:.4f}, nmi={best_nmi:.4f}, ari={best_ari:.4f}, epoch={best_epoch}")
    print(f"[info] Model saved to {Path(args.output_dir) / 'best_model.pkl'}")
    
    return {
        "best_acc": best_acc,
        "best_nmi": best_nmi,
        "best_ari": best_ari,
        "best_epoch": best_epoch,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DAEGC Finetune (Local Files)")
    parser.add_argument("--dataset", type=str, default="Cora", help="Dataset name")
    parser.add_argument("--data_dir", type=str, default="data/cora", help="Data directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--embedding_size", type=int, default=16)
    parser.add_argument("--n_clusters", type=int, default=7)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--v", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=0.0001)  # 与 GitHub 一致
    parser.add_argument("--weight_decay", type=float, default=5e-3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--update_interval", type=int, default=1)
    parser.add_argument("--w_kl", type=float, default=10.0)  # 与 GitHub 一致
    parser.add_argument("--pretrain_path", type=str, default="outputs/daegc_pretrain_cora/best_model.pkl")
    parser.add_argument("--output_dir", type=str, default="outputs/daegc_finetune_cora")
    
    args = parser.parse_args()
    
    finetune(args.dataset, args)
