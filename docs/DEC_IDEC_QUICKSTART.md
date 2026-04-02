# DEC / IDEC 快速复现教程

> 本教程用于在本地 CPU 环境验证 DEC 和 IDEC 算法实现，验证代码正确性后可在更强算力环境运行完整复现。

---

## 目录

1. [环境准备](#1-环境准备)
2. [快速验证（推荐先行）](#2-快速验证推荐先行)
3. [完整复现参数](#3-完整复现参数)
4. [运行命令](#4-运行命令)
5. [结果解读](#5-结果解读)
6. [常见问题](#6-常见问题)
7. [论文参考指标](#7-论文参考指标)

---

## 1. 环境准备

### 1.1 检查 Python 环境

```bash
# 查看 Python 版本
python3 --version
# 推荐 Python 3.8+

# 检查虚拟环境
ls .venv/bin/python
```

### 1.2 安装依赖

```bash
cd clauter

# 激活虚拟环境
source .venv/bin/activate

# 安装所有依赖
pip install numpy scipy scikit-learn pyyaml torch munkres

# 验证
python -c "import torch; import scipy; print('OK')"
```

### 1.3 确认数据集文件存在

```bash
ls data/mnist_all.mat
ls data/usps_all.mat
```

---

## 2. 快速验证（推荐先行）

> **目的**：用最小代价验证代码逻辑正确，不追求精度。
> **预计时间**：5-15 分钟（CPU）
> **数据集**：MNIST 取 5000 样本子集

### 2.1 生成快速验证配置

创建 `configs/dec_quick.yaml`：

```yaml
# DEC 快速验证配置
# 目的：验证代码能跑通，不追求精度
# 预计时间：~5 分钟（CPU）

seed: 42

dataset:
  name: mnist
  source: mat
  path: data/mnist_all.mat
  project_root: .

model:
  name: dec
  hidden_dims: [256, 128, 64]    # 缩小网络，加速验证
  embed_dim: 10
  num_clusters: 10
  dropout: 0.2

optimizer:
  weight_decay: 0.0

trainer:
  # ========== 快速验证：大幅缩减 ==========
  layer_pretrain_iters: 500         # 论文 50000，缩减 100 倍
  layer_pretrain_lr: 0.1
  layer_pretrain_batch_size: 256
  ae_finetune_iters: 1000           # 论文 100000，缩减 100 倍
  ae_finetune_lr: 0.1
  ae_finetune_decay_iters: 500       # 每 500 衰减
  ae_finetune_momentum: 0.9
  clustering_epochs: 50              # 聚类阶段 max epochs
  clustering_lr: 0.01
  target_update_interval: 1
  stop_tol: 0.01                    # 收敛阈值放宽到 1%
  eval_every: 1
  # ========== 快速验证参数 end ==========

output:
  dir: outputs/dec_quick
```

创建 `configs/idec_quick.yaml`：

```yaml
# IDEC 快速验证配置
# 目的：验证代码能跑通，不追求精度
# 预计时间：~5 分钟（CPU）

seed: 42

dataset:
  name: mnist
  source: mat
  path: data/mnist_all.mat
  project_root: .

model:
  name: idec
  hidden_dims: [256, 128, 64]    # 缩小网络
  embed_dim: 10
  num_clusters: 10
  gamma: 0.1                     # IDEC 核心参数

optimizer:
  weight_decay: 0.0

trainer:
  layer_pretrain_iters: 500
  layer_pretrain_lr: 0.1
  layer_pretrain_batch_size: 256
  ae_finetune_iters: 1000
  ae_finetune_lr: 0.001           # IDEC MNIST 论文用 Adam lr=0.001
  ae_finetune_decay_iters: 500
  clustering_epochs: 50
  clustering_lr: 0.001             # IDEC MNIST 论文
  clustering_optimizer: adam
  clustering_momentum: 0.9
  target_update_interval: 10       # 每 10 步更新一次（加速）
  stop_tol: 0.01
  eval_every: 1

output:
  dir: outputs/idec_quick
```

### 2.2 运行快速验证

```bash
cd clauter
source .venv/bin/activate

# 先跑 DEC（更快，因为聚类阶段无 decoder）
python scripts/train_dec_idec.py --config configs/dec_quick.yaml

# 再跑 IDEC
python scripts/train_dec_idec.py --config configs/idec_quick.yaml
```

### 2.3 快速验证成功标准

预期输出：
```
=== Phase 1: Layer-wise Pretrain ===
  Layer 1/4 Epoch 500/500: recon_loss=0.xxxx
  Layer 2/4 Epoch 500/500: recon_loss=0.xxxx
  ...
=== Phase 2: End-to-End Pretrain ===
  Epoch 1000/1000: recon_loss=0.xxxx
=== Phase 3: Clustering Finetune ===
  [P-update @10] acc=0.xxxx nmi=0.xxxx ari=0.xxxx
  [P-update @20] acc=0.xxxx nmi=0.xxxx ari=0.xxxx
  Final metrics: acc=0.xxxx nmi=0.xxxx ari=0.xxxx
[DONE] Results saved to outputs/dec_quick
```

**关键检查点**：
- [x] Phase 1 无报错，逐层 loss 下降
- [x] Phase 2 无报错，重构 loss 下降
- [x] Phase 3 聚类指标出现（ACC > 0.1 即表示代码逻辑正常）
- [x] 输出文件 `best_metrics.json` 存在

---

## 3. 完整复现参数

> **目的**：严格复现论文结果
> **预计时间**：~2 小时（CPU，4核）

### 3.1 DEC + MNIST

```bash
python scripts/train_dec_idec.py --config configs/dec_mnist.yaml
```

预期结果：ACC ≈ 84%，NMI ≈ 84%

### 3.2 IDEC + MNIST

```bash
python scripts/train_dec_idec.py --config configs/idec_mnist.yaml
```

预期结果：ACC ≈ 88%，NMI ≈ 87%

### 3.3 DEC + USPS

```bash
python scripts/train_dec_idec.py --config configs/dec_usps.yaml
```

### 3.4 IDEC + USPS

```bash
python scripts/train_dec_idec.py --config configs/idec_usps.yaml
```

预期结果：ACC ≈ 76%，NMI ≈ 78%

---

## 4. 运行命令

### 4.1 基本命令

```bash
cd clauter
source .venv/bin/activate

# 运行任意配置
python scripts/train_dec_idec.py --config configs/你的配置文件.yaml
```

### 4.2 查看输出

```bash
# 查看训练历史
cat outputs/dec_quick/history.csv

# 查看最佳指标
cat outputs/dec_quick/best_metrics.json

# 查看最终指标
cat outputs/dec_quick/final_metrics.json
```

### 4.3 监控训练进度

训练过程中会打印每个阶段进度：

```
=== Phase 1: Layer-wise Pretrain ===
  Layer 1/4 Epoch 500/500: recon_loss=0.0231   ← 实时重构损失
  Layer 2/4 Epoch 500/500: recon_loss=0.0189
  Layer 3/4 Epoch 500/500: recon_loss=0.0152
  Layer 4/4 Epoch 500/500: recon_loss=0.0121

=== Phase 2: End-to-End Pretrain ===
  Epoch 1000/1000: recon_loss=0.0087           ← 逐 epoch 显示

=== Phase 3: Clustering Finetune ===
  Initialized 10 cluster centers from KMeans (ACC=0.xxxx)
  [P-update @1] acc=0.xxxx nmi=0.xxxx ari=0.xxxx label_change=0.xxxx
  [P-update @2] acc=0.xxxx nmi=0.xxxx ari=0.xxxx label_change=0.xxxx
  ...
  Converged: label_change=0.0008 < tol=0.001   ← 收敛即停止
```

### 4.4 中断和恢复

训练随时可 `Ctrl+C` 中断，下次运行会覆盖输出（不支持断点续训）。

如需保存中间状态，可手动复制输出目录：
```bash
cp -r outputs/dec_quick outputs/dec_quick_backup_$(date +%H%M%S)
```

---

## 5. 结果解读

### 5.1 输出文件说明

| 文件 | 内容 |
|------|------|
| `history.csv` | 每个评估节点的 loss 和指标 |
| `best_metrics.json` | 最佳 ACC/NMI/ARI 及对应 epoch |
| `final_metrics.json` | 最终指标 |
| `last.ckpt` | 模型权重（AE 预训练权重） |
| `pretrain_ae.pkl` | Phase 2 结束时的 AE 权重 |

### 5.2 评估指标说明

| 指标 | 含义 | 范围 | 论文参考 |
|------|------|------|---------|
| **ACC** | 聚类准确率（最优标签映射后） | 0-1 | DEC: 84.30%, IDEC: 88.06% |
| **NMI** | 归一化互信息 | 0-1 | DEC: 83.72%, IDEC: 86.72% |
| **ARI** | 调整兰德指数 | -1-1 | 值越高越好 |

### 5.3 验收标准

代码逻辑验证成功 = Phase 3 输出有 ACC/NMI/ARI 数值（非 NaN）

完整复现验收标准：

| 算法 | 数据集 | ACC 达标 | NMI 达标 |
|------|--------|---------|---------|
| DEC | MNIST | ≥ 80% | ≥ 80% |
| IDEC | MNIST | ≥ 85% | ≥ 83% |
| DEC | USPS | ≥ 70% | ≥ 72% |
| IDEC | USPS | ≥ 74% | ≥ 76% |

---

## 6. 常见问题

### Q1: Phase 1 或 Phase 2 loss 不下降

**原因**：学习率过大或过小，或网络结构问题

**排查**：
```python
# 检查配置文件中的 lr 是否合理
layer_pretrain_lr: 0.1      # 应在 0.01-0.2 之间
ae_finetune_lr: 0.1          # DEC 用 0.1，IDEC MNIST 用 0.001
```

### Q2: Phase 3 ACC 极低（< 0.15）

**原因**：预训练不充分（iterations 太少）或收敛阈值设置不当

**解决**：增加 Phase 1/2 迭代次数，或放宽 `stop_tol`

### Q3: 内存不足（OOM）

**原因**：数据集过大 + batch_size 过大

**解决**：
```yaml
# 在配置文件中减小 batch_size
layer_pretrain_batch_size: 128   # 从 256 减半
ae_finetune_batch_size: 128
```

### Q4: ImportError 或 ModuleNotFoundError

**原因**：缺少依赖包

**解决**：
```bash
source .venv/bin/activate
pip install numpy scipy scikit-learn pyyaml torch munkres
```

### Q5: 收敛太慢或永不收敛

**原因**：`target_update_interval` 设置过大，或 `stop_tol` 过小

**解决**：
```yaml
target_update_interval: 1      # 每步更新 P（DEC 默认）
target_update_interval: 10     # IDEC 可适当放大
stop_tol: 0.001               # 0.1% 标签变化即停止
```

### Q6: 运行时 CPU 被占满导致系统卡顿

**原因**：PyTorch 默认使用所有 CPU 核心

**解决**：限制 CPU 线程数
```bash
# 运行前设置环境变量
export OMP_NUM_THREADS=2      # 只用 2 个核心
export MKL_NUM_THREADS=2
python scripts/train_dec_idec.py --config configs/dec_quick.yaml
```

---

## 7. 论文参考指标

### DEC (Xie et al., ICML 2016)

| 数据集 | ACC | NMI |
|--------|-----|-----|
| MNIST | **84.30%** | **83.72%** |
| STL-HOG | 35.90% | - |
| REUTERS-10K | 72.17% | - |

### IDEC (Guo et al., IJCAI 2017)

| 数据集 | ACC | NMI |
|--------|-----|-----|
| MNIST | **88.06%** | **86.72%** |
| USPS | **76.05%** | **78.46%** |
| REUTERS-10K | 75.64% | 49.81% |

### 与传统方法对比（论文原文）

| 方法 | MNIST ACC |
|------|-----------|
| k-means | 53.49% |
| AE + k-means | 81.84% |
| DEC | 84.30% |
| IDEC | **88.06%** |

---

## 快速检查清单

```
[ ] 虚拟环境已激活 (.venv)
[ ] 依赖已安装 (torch, scipy, scikit-learn, numpy, pyyaml, munkres)
[ ] 数据集文件存在 (mnist_all.mat, usps_all.mat)
[ ] dec_quick.yaml / idec_quick.yaml 已创建
[ ] 运行 python scripts/train_dec_idec.py --config configs/dec_quick.yaml
[ ] 输出无报错，Phase 3 有 ACC/NMI/ARI 数值
[ ] best_metrics.json 文件生成
```

---

*本文档由 clauter 深度聚类复现框架生成*
*最后更新：2026年4月*
