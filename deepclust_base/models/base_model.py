from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict

import torch
import torch.nn as nn


class BaseClusteringModel(nn.Module, ABC):
    @abstractmethod
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def compute_losses(self, outputs: Dict[str, torch.Tensor], x: torch.Tensor, adj: torch.Tensor, cfg: dict) -> Dict[str, torch.Tensor]:
        raise NotImplementedError
