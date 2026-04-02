from __future__ import annotations

import json
from pathlib import Path

import yaml


def load_config(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")

    if p.suffix.lower() in {".yaml", ".yml"}:
        with p.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    if p.suffix.lower() == ".json":
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)

    raise ValueError("Config format not supported. Use .yaml/.yml or .json")


def save_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
