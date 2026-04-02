"""
IDEC (Improved Deep Embedded Clustering) 完整实现
论文: Improved Deep Embedded Clustering with Local Structure Preservation (IJCAI 2017)
Guo, Gao, Liu, Yin

论文核心要点 (与 DEC 的关键区别):
1. 保留 decoder，全程使用 under-complete AE
2. 损失函数: L = L_r + γ * L_c
   - L_r = Σ_i ||x_i - g_W'(z_i)||^2  (MSE 重构损失)
   - L_c = KL(P||Q)                    (DEC 聚类损失)
3. γ = 0.1 (论文 grid search 确定的最佳值)
4. 聚类中心 μ_j 参与优化（与 DEC 一致）
5. 目标分布 P 每 T 次迭代更新一次（而非每个 batch）
   - MNIST: T=140,  USPS: T=30,  REUTERS: T=3
6. 停止条件: 标签变化 < δ = 0.1%
7. 与 DEC 的本质区别:
   - DEC 聚类微调时丢弃 decoder
   - IDEC 保留 decoder + 重构损失 → 保持局部结构

网络结构 (论文):
  Encoder: d → 500 → 500 → 2000 → 10  (ReLU, 除 embedding 层)
  Decoder: 10 → 2000 → 500 → 500 → d  (ReLU, 除输入/输出层)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


# ---------------------------------------------------------------------------
# AutoEncoder (under-complete, 无去噪, 全程保留 decoder)
# ---------------------------------------------------------------------------

class UndercompleteAE(nn.Module):
    """
    Under-complete Autoencoder，全程用于聚类微调。

    特点:
    - 无 Dropout（与 DEC 逐层预训练不同）
    - 无噪声注入（clean data 重构）
    - 保留 decoder，全程参与优化
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Optional[List[int]] = None,
        embedding_dim: int = 10,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [500, 500, 2000]

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.embedding_dim = embedding_dim

        # Encoder
        dims = [input_dim] + hidden_dims + [embedding_dim]
        enc_layers = []
        for i in range(len(dims) - 1):
            enc_layers.append(nn.Linear(dims[i], dims[i + 1]))
            # embedding 层不加 ReLU
            if i < len(dims) - 2:
                enc_layers.append(nn.ReLU(inplace=True))
        self.encoder = nn.Sequential(*enc_layers)

        # Decoder (镜像)
        rev_dims = list(reversed(dims))
        dec_layers = []
        for i in range(len(rev_dims) - 1):
            dec_layers.append(nn.Linear(rev_dims[i], rev_dims[i + 1]))
            # 第一层(重建原始数据) 不激活，其他 ReLU
            if i > 0:
                dec_layers.append(nn.ReLU(inplace=True))
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        x_recon = self.decode(z)
        return z, x_recon


# ---------------------------------------------------------------------------
# 聚类核心（与 DEC 相同）
# ---------------------------------------------------------------------------

class IDECClusteringLayer(nn.Module):
    """可学习的聚类中心 (同 DEC，但 α 固定为 1)"""

    def __init__(self, num_clusters: int, embedding_dim: int):
        super().__init__()
        self.centers = Parameter(torch.Tensor(num_clusters, embedding_dim))
        nn.init.xavier_normal_(self.centers.data)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Student-t 软分配 (α=1):
        q_ij = (1 + ||z_i - μ_j||^2)^(-1) / Σ_j' (1 + ||z_i - μ_j'||^2)^(-1)
        """
        dist_sq = torch.sum(
            (z.unsqueeze(1) - self.centers.unsqueeze(0)) ** 2, dim=2
        )
        q = 1.0 / (1.0 + dist_sq)
        q = (q.t() / torch.clamp(q.sum(dim=1), min=1e-12)).t()
        return q


def compute_target_distribution(q: torch.Tensor) -> torch.Tensor:
    """
    论文公式 (4):
    p_ij = q_ij^2 / f_j / Σ_j' q_ij'^2 / f_j'
    其中 f_j = Σ_i q_ij
    """
    weight = q ** 2 / torch.clamp(q.sum(dim=0, keepdim=True), min=1e-12)
    return weight / torch.clamp(weight.sum(dim=1, keepdim=True), min=1e-12)


# ---------------------------------------------------------------------------
# IDEC 完整模型
# ---------------------------------------------------------------------------

class IDEC(nn.Module):
    """
    Improved Deep Embedded Clustering

    核心: 保留 AE decoder，重构损失约束嵌入空间不被聚类损失破坏。

    三阶段:
    1. 逐层贪婪预训练 (layer-wise pretrain): 每层作为 AE 训练
    2. 端到端精调 (end-to-end pretrain): 整个 AE 微调，重建 loss
    3. 聚类微调 (clustering finetune): L = L_r + γ * L_c，decoder 参与优化

    与 DEC 的区别: 聚类阶段 decoder 保留 + 重构损失 + γ 控制扭曲程度
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Optional[List[int]] = None,
        embedding_dim: int = 10,
        num_clusters: int = 10,
        gamma: float = 0.1,       # 聚类损失系数 (论文推荐 0.1)
        pretrain_path: Optional[str] = None,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [500, 500, 2000]

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.embedding_dim = embedding_dim
        self.num_clusters = num_clusters
        self.gamma = gamma

        self.ae = UndercompleteAE(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            embedding_dim=embedding_dim,
        )

        self.cluster_layer = IDECClusteringLayer(num_clusters, embedding_dim)

        self.training_stage = "pretrain_layerwise"

        if pretrain_path:
            self.load_pretrain(pretrain_path)

    def set_stage(self, stage: str) -> None:
        if stage not in {"pretrain_layerwise", "pretrain_finetune", "clustering"}:
            raise ValueError(f"Unknown stage: {stage}")
        self.training_stage = stage

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.ae.encode(x)

    def get_Q(self, z: torch.Tensor) -> torch.Tensor:
        return self.cluster_layer(z)

    def init_centers_from_embeddings(
        self, z: torch.Tensor, seed: int = 42
    ) -> Dict:
        """用 KMeans 初始化聚类中心"""
        from sklearn.cluster import KMeans

        z_np = z.detach().cpu().numpy()
        km = KMeans(n_clusters=self.num_clusters, n_init=20, random_state=seed)
        y_pred = km.fit_predict(z_np)
        self.cluster_layer.centers.data = torch.tensor(
            km.cluster_centers_, dtype=torch.float32, device=z.device
        )
        return {"y_pred": y_pred, "centers": km.cluster_centers_}

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        接口兼容，adj 仅作占位。
        输出:
          z: 嵌入向量
          q: 软分配概率
          x_recon: 重构 (仅非聚类阶段)
        """
        z = self.ae.encode(x)
        q = self.cluster_layer(z)

        outputs: Dict[str, torch.Tensor] = {"z": z, "q": q}

        if self.training_stage != "clustering":
            x_recon = self.ae.decode(z)
            outputs["x_recon"] = x_recon

        return outputs

    def compute_losses(
        self,
        outputs: Dict[str, torch.Tensor],
        x: torch.Tensor,
        adj: torch.Tensor,
        cfg: dict,
    ) -> Dict[str, torch.Tensor]:
        """统一损失接口"""

        if self.training_stage == "pretrain_layerwise":
            raise RuntimeError("逐层预训练损失应在外部计算")

        elif self.training_stage == "pretrain_finetune":
            # 端到端精调：纯重构损失
            loss_rec = F.mse_loss(outputs["x_recon"], x)
            return {"loss_total": loss_rec, "loss_rec": loss_rec}

        else:  # clustering
            # IDEC 核心: L = L_r + γ * L_c
            x_recon = outputs["x_recon"]
            q = outputs["q"]
            p = compute_target_distribution(q.detach())

            loss_rec = F.mse_loss(x_recon, x)
            loss_kl = F.kl_div(q.log(), p, reduction="batchmean")
            loss_total = loss_rec + self.gamma * loss_kl

            return {
                "loss_total": loss_total,
                "loss_rec": loss_rec,
                "loss_kl": loss_kl,
            }

    # ------------------------------------------------------------------
    # 聚类微调阶段: 单独更新聚类中心
    # 论文公式 (9)(10)(11)(12)(13) 在 train_dec_idec.py 中实现
    # 这里提供辅助方法
    # ------------------------------------------------------------------

    def update_cluster_centers(self, z: torch.Tensor, q: torch.Tensor, p: torch.Tensor, lr: float):
        """
        论文公式 (10):
        ∂L_c/∂μ_j = (2/m) * Σ_i [(1+||z_i-μ_j||^2)^(-1) * (q_ij - p_ij) * (z_i - μ_j)]
        μ_j_new = μ_j - lr * ∂L_c/∂μ_j
        """
        dist_sq_term = 1.0 / torch.clamp(
            1.0 + torch.sum((z.unsqueeze(1) - self.cluster_layer.centers.unsqueeze(0)) ** 2, dim=2),
            min=1e-12,
        )
        diff = z.unsqueeze(1) - self.cluster_layer.centers.unsqueeze(0)  # (N, K, d)
        grad = (
            2
            * torch.sum(
                dist_sq_term.unsqueeze(2) * (q.unsqueeze(2) - p.unsqueeze(2)) * diff,
                dim=0,
            )
            / z.shape[0]
        )
        with torch.no_grad():
            self.cluster_layer.centers.sub_(lr * grad)

    # ------------------------------------------------------------------
    # 逐层预训练 (与 DEC 完全相同，复制用于独立使用)
    # ------------------------------------------------------------------

    def pretrain_layerwise(
        self,
        x: torch.Tensor,
        epochs_per_layer: int = 50,
        lr: float = 0.1,
        corruption_level: float = 0.2,
        verbose: bool = True,
    ) -> None:
        """
        论文 §3.1: 逐层贪婪预训练 stacked denoising AE
        """
        import numpy as np

        np.random.seed(42)
        torch.manual_seed(42)

        all_dims = [self.input_dim] + self.hidden_dims + [self.embedding_dim]
        layer_inputs = [x.detach().clone()]

        for layer_idx in range(len(all_dims) - 1):
            in_dim = all_dims[layer_idx]
            out_dim = all_dims[layer_idx + 1]

            ae_enc = nn.Linear(in_dim, out_dim, bias=True)
            ae_dec = nn.Linear(out_dim, in_dim, bias=True)
            nn.init.xavier_normal_(ae_enc.weight)
            nn.init.xavier_normal_(ae_dec.weight)
            nn.init.zeros_(ae_enc.bias)
            nn.init.zeros_(ae_dec.bias)
            ae_enc = ae_enc.to(x.device)
            ae_dec = ae_dec.to(x.device)

            cur_input = layer_inputs[-1].detach()
            optimizer = torch.optim.Adam(
                list(ae_enc.parameters()) + list(ae_dec.parameters()), lr=lr
            )

            for epoch in range(1, epochs_per_layer + 1):
                # 去噪: 随机置零 corruption_level 比例的维度
                mask = torch.rand_like(cur_input) > corruption_level
                x_corrupted = cur_input * mask.float()

                z = F.relu(ae_enc(x_corrupted)) if layer_idx < len(all_dims) - 2 else ae_enc(x_corrupted)
                use_relu = layer_idx > 0
                x_recon = ae_dec(z)
                if use_relu:
                    x_recon = F.relu(x_recon)

                loss = F.mse_loss(x_recon, cur_input.detach())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if verbose and epoch % 10 == 0:
                    print(
                        f"  [Layer {layer_idx+1}/{len(all_dims)-1}] "
                        f"Epoch {epoch}: loss={loss.item():.6f}"
                    )

            self._copy_layer_weight(layer_idx, ae_enc.weight, ae_enc.bias)
            self._copy_decoder_weight(layer_idx, ae_dec.weight, ae_dec.bias)

            with torch.no_grad():
                next_h = F.relu(ae_enc(cur_input)) if layer_idx < len(all_dims) - 2 else ae_enc(cur_input)
                layer_inputs.append(next_h)

    def _copy_layer_weight(self, layer_idx: int, weight: torch.Tensor, bias: torch.Tensor) -> None:
        enc = self.ae.encoder
        # encoder 是 Linear, ReLU 交替
        linear_idx = layer_idx * 2 if layer_idx > 0 else 0
        linear = enc[linear_idx]
        with torch.no_grad():
            linear.weight.copy_(weight)
            linear.bias.copy_(bias)

    def _copy_decoder_weight(self, layer_idx: int, weight: torch.Tensor, bias: torch.Tensor) -> None:
        dec = self.ae.decoder
        # decoder: Linear, ReLU 交替; 最后一组没有 ReLU
        # 实际是 [Linear, ReLU, Linear, ReLU, Linear] (3组)
        # 层索引反向: layer_idx=0 → dec[-1], layer_idx=1 → dec[-3]
        dec_len = len(dec)
        dec_linear_idx = dec_len - 1 - (layer_idx * 2)
        dec_linear_idx = max(0, min(dec_linear_idx, dec_len - 1))
        if dec_linear_idx < dec_len and hasattr(dec[dec_linear_idx], 'weight'):
            linear = dec[dec_linear_idx]
            with torch.no_grad():
                linear.weight.copy_(weight)
                linear.bias.copy_(bias)

    def load_pretrain(self, path: str) -> None:
        state = torch.load(path, map_location="cpu", weights_only=True)
        self.ae.load_state_dict(state, strict=False)
        print(f"[IDEC] Loaded pretrain weights from {path}")


# ---------------------------------------------------------------------------
# 独立损失函数 (可选，供外部调用)
# ---------------------------------------------------------------------------

def idec_clustering_loss(
    q: torch.Tensor, p: torch.Tensor, x_recon: torch.Tensor, x: torch.Tensor, gamma: float = 0.1
) -> Dict[str, torch.Tensor]:
    loss_rec = F.mse_loss(x_recon, x)
    loss_kl = F.kl_div(q.log(), p.detach(), reduction="batchmean")
    return {
        "loss_total": loss_rec + gamma * loss_kl,
        "loss_rec": loss_rec,
        "loss_kl": loss_kl,
    }
