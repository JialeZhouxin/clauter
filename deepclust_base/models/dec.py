"""
DEC (Deep Embedded Clustering) 完整实现
论文: Unsupervised Deep Embedding for Clustering Analysis (ICML 2016)
Xie, Girshick, Farhadi

论文核心要点:
1. 两阶段: (a) 堆叠去噪自编码器预训练 + (b) KL散度聚类微调
2. 软分配: Student-t 分布 q_ij = (1 + ||z_i - μ_j||^2)^(-1) / Σ_j' (1 + ||z_i - μ_j'||^2)^(-1)
3. 目标分布: p_ij = q_ij^2 / f_j / Σ_j' (q_ij^2 / f_j')
   其中 f_j = Σ_i q_ij
4. α = 1 (Student-t 自由度固定)
5. 聚类微调时 decoder 丢弃，只保留 encoder
6. 优化: SGD + momentum, lr=0.01 (聚类阶段), tol=0.1%
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


# ---------------------------------------------------------------------------
# 网络层
# ---------------------------------------------------------------------------

class DECEncoderLayer(nn.Module):
    """单层去噪自编码器 encoder 块（带 Dropout + ReLU）"""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_dim, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class DECDecoderLayer(nn.Module):
    """单层去噪自编码器 decoder 块（带 Dropout + ReLU）"""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.2, use_relu: bool = True):
        super().__init__()
        layers = [
            nn.Dropout(p=dropout),
            nn.Linear(in_dim, out_dim),
        ]
        if use_relu:
            layers.append(nn.ReLU(inplace=True))
        self.decoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)


class StackedDenoisingAutoEncoder(nn.Module):
    """
    逐层贪婪预训练的堆叠去噪自编码器。

    论文结构 (d-500-500-2000-10):
      MNIST  d=784  → 500 → 500 → 2000 → 10
      USPS   d=256  → 500 → 500 → 2000 → 10
      REUTERS d=2000 → 500 → 500 → 2000 → 10

    第一/二/三层: ReLU
    第四层(embedding层): 线性（不激活，保留全部信息）
    decoder: 镜像结构
    decoder 第一层: 线性（重建原始数据可能有正负值）
    decoder 其他层: ReLU
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        embedding_dim: int,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim

        # Encoder
        all_dims = [input_dim] + hidden_dims + [embedding_dim]
        encoder_layers = []
        for i in range(len(all_dims) - 1):
            in_d, out_d = all_dims[i], all_dims[i + 1]
            # embedding 层不激活
            use_relu = (i < len(all_dims) - 2)
            encoder_layers.append(
                DECEncoderLayer(in_d, out_d, dropout=dropout) if use_relu
                else nn.Linear(in_d, out_d)
            )
        self.encoder = nn.ModuleList(encoder_layers)

        # Decoder (镜像)
        all_rev = list(reversed(all_dims))
        decoder_layers = []
        for i in range(len(all_rev) - 1):
            in_d, out_d = all_rev[i], all_rev[i + 1]
            # decoder 第一层(重建原始数据): 线性
            # decoder 其他层: ReLU
            use_relu = (i > 0)
            decoder_layers.append(
                DECDecoderLayer(in_d, out_d, dropout=dropout, use_relu=use_relu)
            )
        self.decoder = nn.ModuleList(decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.encoder:
            h = layer(h)
        return h

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = z
        for layer in self.decoder:
            h = layer(h)
        return h

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        x_recon = self.decode(z)
        return z, x_recon

    def pretrain_layer(
        self,
        x: torch.Tensor,
        layer_idx: int,
        optimizer,
    ) -> Tuple[torch.Tensor, float]:
        """
        贪婪逐层预训练单个 AE 层。

        Args:
            x: 原始输入数据 (N, input_dim)
            layer_idx: 要训练的层索引 (0-based)
            optimizer: 该层的优化器

        Returns:
            encoded: 该层的输出 (N, layer_out_dim)
            loss: 重构损失
        """
        # 冻结之前的层，用 identity passthrough
        with torch.no_grad():
            h = x
            for i, layer in enumerate(self.encoder[:layer_idx]):
                if isinstance(layer, DECEncoderLayer):
                    # 带 ReLU，直接前向
                    h = layer.encoder[1](layer.encoder[0](h))  # dropout then linear
                else:
                    h = layer(h)
            # 对于 encoder[:layer_idx] 中的 DECEncoderLayer，需要特殊处理
            # 简化：逐层训练时只用到 encoder[layer_idx] 本身
            # 前面的层作为 fixed identity
            pass

        # 实际逐层训练：冻结已训练的层，把当前层作为 AE
        h_fixed = x
        for i in range(layer_idx):
            layer = self.encoder[i]
            if isinstance(layer, DECEncoderLayer):
                # 获取 weight 和 bias
                linear = layer.encoder[1]
                h_fixed = F.relu(F.linear(h_fixed, linear.weight, linear.bias))
            else:
                h_fixed = F.linear(h_fixed, layer.weight, layer.bias)

        # 当前层作为 AE: encode -> decode
        linear_enc = (
            self.encoder[layer_idx].encoder[1]
            if isinstance(self.encoder[layer_idx], DECEncoderLayer)
            else self.encoder[layer_idx]
        )
        z = F.relu(F.linear(h_fixed, linear_enc.weight, linear_enc.bias)) if layer_idx < len(self.encoder) - 1 else F.linear(h_fixed, linear_enc.weight, linear_enc.bias)

        # 构建该层的临时 decoder
        dec_in_dim = z.shape[1]
        dec_out_dim = h_fixed.shape[1]  # 重建到上一层的维度
        dec_linear = nn.Linear(dec_in_dim, dec_out_dim, bias=True).to(x.device)
        x_recon = dec_linear(z)

        optimizer.zero_grad()
        loss = F.mse_loss(x_recon, h_fixed.detach())
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            z_updated = (
                F.relu(F.linear(h_fixed, linear_enc.weight, linear_enc.bias))
                if layer_idx < len(self.encoder) - 1
                else F.linear(h_fixed, linear_enc.weight, linear_enc.bias)
            )

        return z_updated, loss.item()


# ---------------------------------------------------------------------------
# 聚类核心
# ---------------------------------------------------------------------------

class DECClusteringLayer(nn.Module):
    """
    可学习的聚类中心 μ_j ∈ R^d

    梯度 (论文公式, α=1):
      ∂L/∂z_i = 2 * Σ_j [(1+||z_i-μ_j||^2)^(-1) * (p_ij - q_ij) * (z_i - μ_j)]
      ∂L/∂μ_j = 2 * Σ_i [(1+||z_i-μ_j||^2)^(-1) * (q_ij - p_ij) * (z_i - μ_j)]
    """

    def __init__(self, num_clusters: int, embedding_dim: int):
        super().__init__()
        self.centers = Parameter(torch.Tensor(num_clusters, embedding_dim))
        nn.init.xavier_normal_(self.centers.data)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        返回软分配 Q (N, K)
        q_ij = (1 + ||z_i - μ_j||^2)^(-1) / Σ_j' (1 + ||z_i - μ_j'||^2)^(-1)
        论文原文 α=1, 即指数为 (α+1)/2 = 1
        """
        # 实际论文公式等价于 α=1 时 q_ij = 1/(1+dist^2) / normalization
        # 与 α 通用形式的换算: (1+dist^2/α)^(-(α+1)/2) → α=1 时 (1+dist^2)^(-1)
        dist_sq = torch.sum((z.unsqueeze(1) - self.centers.unsqueeze(0)) ** 2, dim=2)
        q = 1.0 / (1.0 + dist_sq)
        q = (q.t() / torch.clamp(q.sum(dim=1), min=1e-12)).t()
        return q


def compute_target_distribution(q: torch.Tensor) -> torch.Tensor:
    """
    论文公式 (3):
    p_ij = (q_ij^2 / f_j) / Σ_j' (q_ij^2 / f_j')
    其中 f_j = Σ_i q_ij
    """
    weight = q ** 2 / torch.clamp(q.sum(dim=0, keepdim=True), min=1e-12)
    return weight / torch.clamp(weight.sum(dim=1, keepdim=True), min=1e-12)


# ---------------------------------------------------------------------------
# DEC 完整模型
# ---------------------------------------------------------------------------

class DEC(nn.Module):
    """
    Deep Embedded Clustering

    三个阶段:
    1. 逐层贪婪预训练 (layer-wise pretrain): 每层作为 AE 训练
    2. 端到端预训练精调 (end-to-end pretrain): 整个 SAE 微调，重建 loss
    3. 聚类微调 (clustering finetune): 丢弃 decoder，只用 encoder + KL loss

    接口兼容 deepclust_base.BaseClusteringModel:
      forward(x, adj) 返回 dict，必须有 q/z/loss_total
      compute_losses() 返回各分项 loss
    注意: DEC 不使用图结构，adj 参数存在但仅作接口兼容
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Optional[List[int]] = None,
        embedding_dim: int = 10,
        num_clusters: int = 10,
        dropout: float = 0.2,
        pretrain_path: Optional[str] = None,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [500, 500, 2000]

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.embedding_dim = embedding_dim
        self.num_clusters = num_clusters
        self.dropout = dropout

        # 主 SAE
        self.sae = StackedDenoisingAutoEncoder(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            embedding_dim=embedding_dim,
            dropout=dropout,
        )

        # 聚类层
        self.cluster_layer = DECClusteringLayer(num_clusters, embedding_dim)

        # 当前阶段: pretrain_layerwise | pretrain_finetune | clustering
        self.training_stage = "pretrain_layerwise"

        if pretrain_path:
            self.load_pretrain(pretrain_path)

    def set_stage(self, stage: str) -> None:
        if stage not in {"pretrain_layerwise", "pretrain_finetune", "clustering"}:
            raise ValueError(f"Unknown stage: {stage}")
        self.training_stage = stage

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """纯编码，不带任何激活或后处理"""
        return self.sae.encode(x)

    def get_Q(self, z: torch.Tensor) -> torch.Tensor:
        """Student-t 软分配"""
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

    def forward(
        self, x: torch.Tensor, adj: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        DEC forward（接口兼容，adj 仅作占位）
        clustering 阶段: 输出 q (软分配) 和 z (嵌入)
        聚类微调阶段保留: z, q
        预训练阶段保留: z, x_recon
        """
        z = self.sae.encode(x)
        q = self.cluster_layer(z)

        outputs = {"z": z, "q": q}

        if self.training_stage != "clustering":
            _, x_recon = self.sae(x)
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
            # 逐层预训练损失在外层计算
            raise RuntimeError(
                "逐层预训练损失应在外部单独计算，不走此接口"
            )

        elif self.training_stage == "pretrain_finetune":
            # 端到端精调：纯重构损失
            x_recon = outputs["x_recon"]
            loss_rec = F.mse_loss(x_recon, x)
            return {"loss_total": loss_rec, "loss_rec": loss_rec}

        else:  # clustering
            # 聚类微调：KL(Q||P)
            q = outputs["q"]
            p = compute_target_distribution(q.detach())
            loss_kl = F.kl_div(q.log(), p, reduction="batchmean")
            return {"loss_total": loss_kl, "loss_kl": loss_kl}

    # ------------------------------------------------------------------
    # 预训练工具
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
        论文 §3.2 逐层贪婪预训练
        每层作为去噪 AE: encode → decode → MSE
        每个编码器层的激活: ReLU (除最后一层)
        每个解码器层的激活: ReLU (除第一层，重建原始数据)
        """
        import numpy as np

        np.random.seed(42)
        torch.manual_seed(42)

        # 逐层获取中间表示作为伪输入
        h = x.detach().clone()
        layer_inputs = [x]

        all_dims = [self.input_dim] + self.hidden_dims + [self.embedding_dim]

        for layer_idx in range(len(all_dims) - 1):
            in_dim = all_dims[layer_idx]
            out_dim = all_dims[layer_idx + 1]

            # 该层的临时 AE
            ae_enc = nn.Linear(in_dim, out_dim, bias=True)
            ae_dec = nn.Linear(out_dim, in_dim, bias=True)
            nn.init.xavier_normal_(ae_enc.weight)
            nn.init.xavier_normal_(ae_dec.weight)
            nn.init.zeros_(ae_enc.bias)
            nn.init.zeros_(ae_dec.bias)

            ae_enc = ae_enc.to(x.device)
            ae_dec = ae_dec.to(x.device)

            # 当前层伪输入
            cur_input = layer_inputs[-1].detach()

            optimizer = torch.optim.Adam(
                list(ae_enc.parameters()) + list(ae_dec.parameters()), lr=lr
            )

            for epoch in range(1, epochs_per_layer + 1):
                # 去噪: 随机置零 corruption_level 的维度
                mask = torch.rand_like(cur_input) > corruption_level
                x_corrupted = cur_input * mask.float()

                # Encode
                z = F.relu(ae_enc(x_corrupted)) if layer_idx < len(all_dims) - 2 else ae_enc(x_corrupted)

                # Decode
                use_relu = (layer_idx > 0)  # decoder 第一层(重建)不激活
                x_recon = ae_dec(z)
                if use_relu:
                    x_recon = F.relu(x_recon)

                # MSE 重构
                loss = F.mse_loss(x_recon, cur_input.detach())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if verbose and epoch % 10 == 0:
                    print(f"  [Layer {layer_idx+1}/{len(all_dims)-1}] Epoch {epoch}: loss={loss.item():.6f}")

            # 把学好的 encoder weight 迁移到 SAE
            self._copy_layer_weight(
                layer_idx, ae_enc.weight, ae_enc.bias
            )
            self._copy_decoder_weight(
                layer_idx, ae_dec.weight, ae_dec.bias
            )

            # 计算下一层输入
            with torch.no_grad():
                next_h = F.relu(ae_enc(cur_input)) if layer_idx < len(all_dims) - 2 else ae_enc(cur_input)
                layer_inputs.append(next_h)

    def _copy_layer_weight(self, layer_idx: int, weight: torch.Tensor, bias: torch.Tensor) -> None:
        """将预训练好的 weight 复制到 SAE encoder"""
        layer = self.sae.encoder[layer_idx]
        if isinstance(layer, DECEncoderLayer):
            linear = layer.encoder[1]
        else:
            linear = layer
        with torch.no_grad():
            linear.weight.copy_(weight)
            linear.bias.copy_(bias)

    def _copy_decoder_weight(self, layer_idx: int, weight: torch.Tensor, bias: torch.Tensor) -> None:
        """将预训练好的 decoder weight 复制到 SAE decoder"""
        # decoder 是反过来的索引
        dec_idx = len(self.hidden_dims) - layer_idx
        layer = self.sae.decoder[dec_idx]
        if isinstance(layer, DECDecoderLayer):
            linear = layer.decoder[1]
        else:
            linear = layer
        with torch.no_grad():
            linear.weight.copy_(weight)
            linear.bias.copy_(bias)

    def load_pretrain(self, path: str) -> None:
        """加载预训练权重"""
        state = torch.load(path, map_location="cpu", weights_only=True)
        self.sae.load_state_dict(state, strict=False)
        print(f"[DEC] Loaded pretrain weights from {path}")


def dec_pretrain_loss(outputs: Dict[str, torch.Tensor], x: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(outputs["x_recon"], x)


def dec_clustering_loss(q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    return F.kl_div(q.log(), p.detach(), reduction="batchmean")
