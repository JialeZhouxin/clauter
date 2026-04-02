# DAEGC 使用教程

本文档说明如何在 `clauter` 仓库中一行命令运行 DAEGC，并查看 ACC / NMI / ARI 等指标。

## 1. 环境准备

本项目使用本地虚拟环境 `.venv`，推荐始终使用它的 Python：

```bash
cd /home/zjl/clauter
.venv/bin/python --version
```

如果你还没有 `.venv`，可先创建并安装依赖：

```bash
python3 -m venv .venv
.venv/bin/python -m ensurepip --upgrade
.venv/bin/python -m pip install numpy scipy scikit-learn pyyaml torch munkres
```

## 2. 一行命令直接运行并得到指标

论文配置：

```bash
.venv/bin/python scripts/train.py --config configs/daegc_cora.yaml
```

高分调优配置：

```bash
.venv/bin/python scripts/train.py --config configs/daegc_cora_best.yaml
```

运行完成后，终端会直接打印类似下面的结果：

```text
[done] best_metrics={'nmi': ..., 'ari': ..., 'acc': ...}, best_epoch=...
```

## 3. 指标文件在哪

训练完成后，结果会写入输出目录：

- 论文配置：`outputs/daegc_cora/`
- 调优配置：`outputs/daegc_cora_grid/`

常用文件：

- `best_metrics.json`：最终最佳 ACC / NMI / ARI
- `history.csv`：每个 epoch 的训练记录
- `best_model.pkl`：最佳模型权重
- `last.ckpt`：最后一次 checkpoint

## 4. 直接读取指标

如果你只想读取已经保存好的指标：

```bash
.venv/bin/python -c "import json; print(json.load(open('outputs/daegc_cora_grid/best_metrics.json')))"
```

如果你要更规整一点：

```bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path
p = Path('outputs/daegc_cora_grid/best_metrics.json')
print(json.dumps(json.loads(p.read_text()), indent=2, ensure_ascii=False))
PY
```

## 5. 当前推荐的命令

- 想对齐论文：
  - `configs/daegc_cora.yaml`
- 想拿当前最优分数：
  - `configs/daegc_cora_best.yaml`

## 6. 常见注意事项

1. 不要直接用系统 `python3`，它可能没有 torch。
2. 如果出现 CUDA driver 警告，但日志里显示 `device=cpu`，可以忽略。
3. 结果对 finetune 细节非常敏感，尤其是：
   - `target_update_interval`
   - `finetune_eval_mode`
   - `finetune_stop_tol`

## 7. 当前已知结果

- 论文配置：ACC 0.6784，NMI 0.5216，ARI 0.4377
- 调优配置：ACC 0.7360，NMI 0.5478，ARI 0.5368

这说明 DAEGC 对 finetune 动力学非常敏感，单纯照搬超参数不一定等于最优表现。
