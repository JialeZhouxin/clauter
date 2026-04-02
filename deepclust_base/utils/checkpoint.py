from __future__ import annotations

from pathlib import Path

import torch


def save_checkpoint(path: Path, model, optimizer, epoch: int, extra: dict | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "extra": extra or {},
    }
    torch.save(payload, path)


def load_checkpoint(path: Path, model, optimizer=None):
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state"])
    if optimizer is not None and "optimizer_state" in payload:
        optimizer.load_state_dict(payload["optimizer_state"])
    return payload.get("epoch", 0), payload.get("extra", {})
