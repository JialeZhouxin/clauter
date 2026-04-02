from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Mapping


def write_history_csv(path: Path, rows: Iterable[Mapping]):
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)

    keys = sorted({k for row in rows for k in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
