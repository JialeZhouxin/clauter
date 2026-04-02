"""
DAEGC (Deep Attentional Embedded Graph Clustering) 完全按论文实现
论文: Attributed Graph Clustering: A Deep Attentional Embedding Approach (IJCAI 2019)

还原要点:
1. GAT 完整注意力机制 (使用 M 矩阵)
2. 两阶段训练: 预训练(GAE) + 聚类微调
3. DEC 软分配 + 目标分布
4. 损失函数: BCE + KL_div
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class GATLayer(nn.Module):
    """完全还原论文中的 GAT 层实现
    
    注意力机制:
    e_ij = leaky_relu(a^T · [W·h_i || W·h_j])
    a_ij = softmax_j(e_ij * M_ij)
    h'_i = ELU(Σ_j a_ij · W·h_j)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        alpha: float = 0.2,
        concat: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        # 特征变换矩阵 W
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        # 注意力向量 a (分量为 self 和 neighbors)
        self.a_self = nn.Parameter(torch.zeros(size=(out_features, 1)))
        nn.init.xavier_uniform_(self.a_self.data, gain=1.414)

        self.a_neighs = nn.Parameter(torch.zeros(size=(out_features, 1)))
        nn.init.xavier_uniform_(self.a_neighs.data, gain=1.414)

        self.leaky_relu = nn.LeakyReLU(self.alpha)

    def forward(
        self, 
        x: torch.Tensor, 
        adj: torch.Tensor, 
        M: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: 节点特征 (N, in_features)
            adj: 邻接矩阵 (N, N), 已归一化
            M: 转移概率矩阵 (N, N), 用于注意力 mask
        Returns:
            输出特征 (N, out_features)
        """
        # 线性变换
        h = torch.mm(x, self.W)  # (N, out_features)

        # 计算注意力分数
        attn_for_self = torch.mm(h, self.a_self)      # (N, 1)
        attn_for_neighs = torch.mm(h, self.a_neighs)  # (N, 1)
        
        # 组合注意力: e_ij = a_self · h_i + a_neighs · h_j
        attn_dense = attn_for_self + torch.transpose(attn_for_neighs, 0, 1)
        
        # 应用 M mask 并 leaky_relu
        attn_dense = torch.mul(attn_dense, M)
        attn_dense = self.leaky_relu(attn_dense)  # (N, N)

        # Mask 零位置
        zero_vec = -9e15 * torch.ones_like(adj)
        adj_masked = torch.where(adj > 0, attn_dense, zero_vec)

        # Softmax 归一化
        attention = F.softmax(adj_masked, dim=1)

        # 聚合邻居信息
        h_prime = torch.matmul(attention, h)  # (N, out_features)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return f"{self.__class__.__name__}({self.in_features} -> {self.out_features})"


class GATEncoder(nn.Module):
    """GAT 编码器 (用于预训练和微调)
    
    结构: input -> GAT -> GAT -> L2 normalize -> z
    """

    def __init__(
        self,
        num_features: int,
        hidden_size: int,
        embedding_size: int,
        alpha: float = 0.2,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.alpha = alpha

        self.conv1 = GATLayer(num_features, hidden_size, alpha)
        self.conv2 = GATLayer(hidden_size, embedding_size, alpha)

    def forward(self, x: torch.Tensor, adj: torch.Tensor, M: torch.Tensor):
        """
        Args:
            x: 节点特征 (N, num_features)
            adj: 邻接矩阵 (N, N)
            M: 转移矩阵 (N, N)
        Returns:
            A_pred: 重建的邻接矩阵 (N, N)
            z: 嵌入向量 (N, embedding_size), L2 normalized
        """
        h = self.conv1(x, adj, M)
        h = self.conv2(h, adj, M)
        
        # L2 normalize (论文原文)
        z = F.normalize(h, p=2, dim=1)
        
        # 内积解码器
        A_pred = self.dot_product_decode(z)
        
        return A_pred, z

    def dot_product_decode(self, z: torch.Tensor) -> torch.Tensor:
        """内积解码器: A_pred = sigmoid(z · z^T)"""
        A_pred = torch.sigmoid(torch.matmul(z, z.t()))
        return A_pred


def target_distribution(q: torch.Tensor) -> torch.Tensor:
    """DEC 目标分布 (论文公式)
    
    p_ij = q_ij^2 / Σ_k q_kj  /  Σ_k (q_ik^2 / Σ_k q_kj)
    """
    weight = q ** 2 / torch.clamp(q.sum(dim=0, keepdim=True), min=1e-12)
    return weight / torch.clamp(weight.sum(dim=1, keepdim=True), min=1e-12)


class DAEGC(nn.Module):
    """DAEGC 完整模型（论文两阶段流程对照版）

    支持：
    1. 预训练阶段：GAE 图重建
    2. 微调阶段：DEC self-training + 图重建
    3. 可选在 forward 中自动从 adj 计算 M
    """

    def __init__(
        self,
        num_features: int,
        hidden_size: int,
        embedding_size: int,
        num_clusters: int,
        alpha: float = 0.2,
        v: float = 1.0,
        pretrain_path: Optional[str] = None,
    ):
        super().__init__()
        self.num_clusters = num_clusters
        self.v = v
        self.training_stage = "pretrain"

        self.gat = GATEncoder(num_features, hidden_size, embedding_size, alpha)

        if pretrain_path is not None:
            self.gat.load_state_dict(torch.load(pretrain_path, map_location="cpu"))
            print(f"Loaded pretrained weights from {pretrain_path}")

        self.cluster_layer = Parameter(torch.Tensor(num_clusters, embedding_size))
        nn.init.xavier_normal_(self.cluster_layer.data)

    def set_stage(self, stage: str) -> None:
        if stage not in {"pretrain", "finetune"}:
            raise ValueError(f"Unknown training stage: {stage}")
        self.training_stage = stage

    @staticmethod
    def _compute_M(adj: torch.Tensor, t: int = 2) -> torch.Tensor:
        """按论文/官方实现从邻接矩阵计算转移矩阵 M。"""
        from sklearn.preprocessing import normalize
        import numpy as np

        adj_np = adj.detach().cpu().numpy()
        tran_prob = normalize(adj_np, norm="l1", axis=0)
        M_np = sum([np.linalg.matrix_power(tran_prob, i) for i in range(1, t + 1)]) / t
        return torch.from_numpy(M_np).to(dtype=torch.float32, device=adj.device)

    def get_Q(self, z: torch.Tensor) -> torch.Tensor:
        """DEC 软分配 (Student-t 分布)"""
        dist_sq = torch.sum((z.unsqueeze(1) - self.cluster_layer.unsqueeze(0)) ** 2, dim=2)
        q = 1.0 / (1.0 + dist_sq / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.clamp(q.sum(dim=1), min=1e-12)).t()
        return q

    def init_cluster_layer_from_embeddings(self, z: torch.Tensor, seed: int = 42) -> dict:
        """用 KMeans 初始化聚类中心；返回初始化聚类结果。"""
        from sklearn.cluster import KMeans

        z_np = z.detach().cpu().numpy()
        kmeans = KMeans(n_clusters=self.num_clusters, n_init=20, random_state=seed)
        y_pred = kmeans.fit_predict(z_np)
        self.cluster_layer.data = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32, device=z.device)
        return {"y_pred": y_pred, "centers": kmeans.cluster_centers_}

    def encode(self, x: torch.Tensor, adj: torch.Tensor, M: torch.Tensor | None = None) -> torch.Tensor:
        """仅编码输出嵌入 z。"""
        if M is None:
            M = self._compute_M(adj)
        _, z = self.gat(x, adj, M)
        return z

    def forward(self, x: torch.Tensor, adj: torch.Tensor, M: torch.Tensor | None = None) -> Dict[str, torch.Tensor]:
        if M is None:
            M = self._compute_M(adj)
        A_pred, z = self.gat(x, adj, M)
        q = self.get_Q(z)
        return {"A_pred": A_pred, "z": z, "q": q}

    def compute_losses(self, outputs: Dict[str, torch.Tensor], x: torch.Tensor, adj: torch.Tensor, cfg: dict) -> Dict[str, torch.Tensor]:
        """统一损失接口：根据训练阶段返回不同损失。"""
        if self.training_stage == "pretrain":
            rec_loss = F.binary_cross_entropy(outputs["A_pred"].view(-1), adj.view(-1))
            return {"loss_total": rec_loss, "loss_rec": rec_loss}

        w_kl = float(cfg.get("w_kl", 10.0))
        q = outputs["q"]
        p = target_distribution(q.detach())
        kl_loss = F.kl_div(q.log(), p, reduction="batchmean")
        rec_loss = F.binary_cross_entropy(outputs["A_pred"].view(-1), adj.view(-1))
        total = w_kl * kl_loss + rec_loss
        return {"loss_total": total, "loss_kl": kl_loss, "loss_rec": rec_loss}


def pretrain_loss(A_pred: torch.Tensor, adj_label: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy(A_pred.view(-1), adj_label.view(-1))


def clustering_loss(Q: torch.Tensor, P: torch.Tensor, A_pred: torch.Tensor, adj_label: torch.Tensor, w_kl: float = 10.0) -> Dict[str, torch.Tensor]:
    kl_loss = F.kl_div(Q.log(), P.detach(), reduction='batchmean')
    rec_loss = F.binary_cross_entropy(A_pred.view(-1), adj_label.view(-1))
    total = w_kl * kl_loss + rec_loss
    return {"loss_total": total, "loss_kl": kl_loss, "loss_rec": rec_loss}
