from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from deepclust_base.datasets import convert_cora_raw_to_npz


def parse_args():
    p = argparse.ArgumentParser(description='Convert raw Cora files to npz format for deepclust_base')
    p.add_argument('--raw-dir', type=str, default='data/cora', help='Directory containing cora.content and cora.cites')
    p.add_argument('--out', type=str, default='data/cora/cora.npz', help='Output npz path')
    return p.parse_args()


def main():
    args = parse_args()
    raw_dir = ROOT / args.raw_dir
    out_path = ROOT / args.out
    meta = convert_cora_raw_to_npz(raw_dir=raw_dir, out_path=out_path)
    print(meta)


if __name__ == '__main__':
    main()
