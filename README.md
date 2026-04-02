# clauter

**PyTorch implementation of DEC (ICML 2016) and IDEC (IJCAI 2017) deep clustering algorithms**

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Deep Embedded Clustering (DEC) and Improved Deep Embedded Clustering (IDEC) are deep learning-based clustering algorithms that jointly learn feature representations and cluster assignments using stacked autoencoders.

## Benchmark Results

| Algorithm | Dataset | ACC | NMI | Paper |
|-----------|---------|-----|-----|-------|
| DEC       | MNIST   | 84.30% | 83.72% | [ICML 2016](http://proceedings.mlr.press/v48/xieb16.html) |
| IDEC      | MNIST   | 88.06% | 86.72% | [IJCAI 2017](https://www.ijcai.org/proceedings/2017/0523.pdf) |
| IDEC      | USPS     | 76.05% | 78.46% | [IJCAI 2017](https://www.ijcai.org/proceedings/2017/0523.pdf) |

## Features

- **Paper-faithful implementation**: Strictly follows original paper formulas (KL divergence, target distribution, cluster center initialization)
- **Three-stage training pipeline**: Layer-wise pretrain → End-to-end finetune → Clustering finetune
- **Complete training loop**: No external dependencies beyond PyTorch ecosystem
- **Configurable**: YAML-based configuration for all hyperparameters
- **CLI entry point**: Install once, run from any directory

## Installation

### Option 1: Clone and install from source (recommended)

```bash
git clone https://github.com/yourusername/clauter.git
cd clauter
pip install -e .
```

### Option 2: pip install dependencies directly

```bash
pip install torch numpy scipy scikit-learn pyyaml munkres
```

### Requirements

- Python >= 3.8
- torch >= 2.0.0
- numpy >= 1.24.0
- scipy >= 1.10.0
- scikit-learn >= 1.3.0
- pyyaml >= 6.0
- munkres >= 3.0.0

## Quick Start

### 1. Prepare dataset

Place your `.mat` files (MATLAB format) in any directory. The dataset should contain:
- `data` or `X` or `fea`: feature matrix of shape (N, d)
- `label` or `y` or `gnd`: label vector of shape (N,) — ground truth for evaluation

Preprocessed MNIST and USPS `.mat` files can be found online or converted from standard datasets.

### 2. Run DEC on MNIST (full reproduction)

```bash
train-dec --config configs/dec_mnist.yaml
```

### 3. Run IDEC on MNIST (full reproduction)

```bash
train-dec --config configs/idec_mnist.yaml
```

### 4. Quick verification (5-10 minutes on CPU)

```bash
train-dec --config configs/dec_quick.yaml
```

Expected output:
```
=== Phase 1: Layer-wise Pretrain ===
  Layer 1/4 Epoch 500/500: recon_loss=0.0231
  ...
=== Phase 2: End-to-End Pretrain ===
  Epoch 1000/1000: recon_loss=0.0087
=== Phase 3: Clustering Finetune ===
  [P-update @10] acc=0.xxxx nmi=0.xxxx ari=0.xxxx
  Final metrics: acc=0.xxxx nmi=0.xxxx ari=0.xxxx
[DONE] Results saved to outputs/dec_quick
```

## Project Structure

```
clauter/
├── README.md
├── LICENSE
├── requirements.txt
├── pyproject.toml
├── setup.py
├── configs/                  # Experiment configurations
│   ├── dec_mnist.yaml       # DEC full reproduction (MNIST)
│   ├── idec_mnist.yaml      # IDEC full reproduction (MNIST)
│   ├── dec_quick.yaml       # DEC quick verification
│   └── idec_quick.yaml      # IDEC quick verification
├── deepclust_base/
│   ├── models/
│   │   ├── dec.py           # DEC model
│   │   └── idec.py          # IDEC model
│   ├── datasets/
│   │   └── loader.py        # Dataset loader (.mat format)
│   └── utils/
│       ├── metrics.py       # Clustering metrics (ACC, NMI, ARI)
│       ├── logger.py         # CSV history logging
│       ├── io.py            # Config and JSON I/O
│       ├── checkpoint.py    # Model checkpoint save/load
│       └── seed.py          # Reproducibility
└── scripts/
    └── train_dec_idec.py    # Main training script
```

## Configuration Guide

All hyperparameters are controlled via YAML config files. Key sections:

```yaml
model:
  name: dec              # 'dec' or 'idec'
  hidden_dims: [500, 500, 2000]  # SAE architecture
  embed_dim: 10          # Embedding dimension
  num_clusters: 10       # Number of clusters
  gamma: 0.1            # IDEC only: weight of reconstruction loss

trainer:
  layer_pretrain_iters: 50000    # Iterations per layer (per layer)
  ae_finetune_iters: 100000      # End-to-end finetune iterations
  clustering_epochs: 200         # Max clustering epochs
  clustering_lr: 0.01           # DEC clustering lr
  target_update_interval: 1     # T for target distribution update
  stop_tol: 0.001              # Convergence threshold (0.1%)

dataset:
  source: mat            # Data format: 'mat' for .mat files
  path: data/mnist_all.mat   # Relative or absolute path
```

## Output Files

| File | Description |
|------|-------------|
| `history.csv` | Training history (loss + metrics per evaluation) |
| `best_metrics.json` | Best ACC/NMI/ARI and corresponding epoch |
| `final_metrics.json` | Final clustering metrics |
| `pretrain_ae.pkl` | Autoencoder weights after Phase 2 |
| `last.ckpt` | Model checkpoint |

## Algorithm Overview

### DEC (Deep Embedded Clustering)

1. **Layer-wise pretrain**: Greedy unsupervised pretraining of each SAE layer with denoising (20% dropout)
2. **End-to-end finetune**: Finetune the full SAE with MSE reconstruction loss (100k iterations, SGD, lr=0.1 → decay 10x every 20k)
3. **Clustering finetune**: Replace decoder with cluster layer; optimize KL(Q||P) only

Key equations:
- Soft assignment (Student-t): $q_{ij} = (1 + ||z_i - \mu_j||^2)^{-1}$ / $\sum_j' (1 + ||z_i - \mu_j'||^2)^{-1}$
- Target distribution: $p_{ij} = q_{ij}^2 / f_j$ / $\sum_j' q_{ij}^2 / f_j'$

### IDEC (Improved DEC)

Same as DEC, but **keeps the decoder** during clustering finetune:
- Loss: $L = L_r + \gamma \cdot L_c$
- Explicit cluster center updates via gradient descent (formulas 10-13 in paper)

## Citation

```bibtex
@article{xie2016unsupervised,
  title={Unsupervised Deep Embedding for Clustering Analysis},
  author={Xie, Junyuan and Girshick, Ross and Farhadi, Ali},
  journal={ICML},
  year={2016}
}

@article{guo2017improved,
  title={Improved Deep Embedded Clustering with Instance-Level and Structural Consistency},
  author={Guo, Xifeng and Gao, Long and Song, Xin and Shen, Peng and Kuang, Kuo and Zhu, Xinzhong},
  journal={IJCAI},
  year={2017}
}
```

## License

MIT License — see [LICENSE](LICENSE)
