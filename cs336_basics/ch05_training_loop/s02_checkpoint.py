"""5.2 检查点保存与恢复。仅接口骨架，待自行实现。"""

from __future__ import annotations

from os import PathLike
from typing import BinaryIO

from torch import nn
from torch.optim import Optimizer


def save_checkpoint(
    model: nn.Module, optimizer: Optimizer, iteration: int,
    out: str | PathLike[str] | BinaryIO,
) -> None:
    raise NotImplementedError

def load_checkpoint(
    src: str | PathLike[str] | BinaryIO, model: nn.Module, optimizer: Optimizer,
) -> int:
    raise NotImplementedError
