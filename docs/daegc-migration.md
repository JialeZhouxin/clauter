# DAEGC 迁移方法

本文档说明如何把旧的 DAEGC 复现方式迁移到当前统一框架 `clauter`。

## 1. 迁移目标

旧方式的核心特点通常是：

- 独立的 pretrain / finetune 脚本
- 手工保存预训练权重
- 单独跑 KMeans 初始化
- 指标与训练流程分散在不同文件中

当前统一框架的目标是：

- 入口统一到 `scripts/train.py`
- 通过 config 自动选择模型与数据集
- 两阶段 DAEGC 流程自动执行
- 训练记录、最佳结果、checkpoint 自动保存

## 2. 从旧流程迁移到新流程

### 2.1 旧流程

常见旧流程类似：

```bash
python scripts/daegc_pretrain.py --epochs 30 --output_dir outputs/daegc_pretrain_cora
python scripts/daegc_finetune.py --pretrain_path outputs/daegc_pretrain_cora/best_model.pkl --epochs 100 --output_dir outputs/daegc_finetune_cora
```

### 2.2 新流程

现在只需要：

```bash
.venv/bin/python scripts/train.py --config configs/daegc_cora.yaml
```

如果要跑当前最优配置：

```bash
.venv/bin/python scripts/train.py --config configs/daegc_cora_best.yaml
```

## 3. 迁移时需要对齐的关键点

### 3.1 环境

- 使用 `.venv/bin/python`
- 确保已安装：`torch`, `numpy`, `scipy`, `scikit-learn`, `pyyaml`, `munkres`
- 默认在 CPU 上运行

### 3.2 数据预处理

对 Cora / 图聚类任务，优先确认：

- `normalize_features`: 是否符合原论文设置
- `normalize_adjacency`: 是否为 `true`
- `add_self_loops`: 是否启用
- `adj_normalize_method`: 是否为 `l1`

### 3.3 模型参数

DAEGC 常用参数：

- `hidden_dim: 256`
- `embed_dim: 16`
- `num_clusters: 7`
- `alpha: 0.2`
- `v: 1.0`

### 3.4 训练协议

两阶段流程必须保留：

1. pretrain
2. KMeans 初始化聚类中心
3. finetune

如果把这三步压成单阶段，结果通常会明显变差。

### 3.5 微调动力学

当前经验表明，DAEGC 的结果对以下参数特别敏感：

- `target_update_interval`
- `finetune_eval_mode`
- `finetune_stop_tol`
- `finetune_epochs`

这也是迁移时最容易“看起来跑了，但结果不对”的地方。

## 4. 旧文件与新文件的对应关系

| 旧习惯 | 新位置 |
|---|---|
| `daegc_pretrain.py` | `scripts/train.py`（自动分发到 DAEGC 两阶段流程） |
| `daegc_finetune.py` | `scripts/train.py`（第二阶段内置） |
| 手工保存最佳模型 | `outputs/*/best_model.pkl` |
| 手工写训练日志 | `outputs/*/history.csv` |
| 手工记录指标 | `outputs/*/best_metrics.json` |

## 5. 建议的迁移步骤

1. 先把旧脚本跑通一次，记下结果。
2. 切到统一入口 `scripts/train.py`。
3. 对齐 config。
4. 跑论文配置。
5. 如果要更好的数值，再试 `configs/daegc_cora_best.yaml`。
6. 将结果写入复现报告。

## 6. 如果要迁移到别的数据集

一般需要改这些地方：

- `dataset.path`
- `model.num_clusters`
- `hidden_dim / embed_dim`
- `pretrain_epochs / finetune_epochs`
- `target_update_interval`
- `output.dir`

如果数据集结构不是 Cora 这种 raw npz，而是原始边表或别的图格式，先写一个数据转换器，把它统一转成 `npz(x, adj, y)` 再接入框架。

## 7. 最后一句话

迁移的重点不是“让脚本能跑”，而是“让训练协议不变”。
如果训练协议变了，结果再好也不算同一个方法。
