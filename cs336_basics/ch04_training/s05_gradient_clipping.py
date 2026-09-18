"""4.5 梯度裁剪。仅接口骨架，待自行实现。"""

from __future__ import annotations

from collections.abc import Iterable

from torch import nn
import torch,math


def gradient_clipping(parameters: Iterable[nn.Parameter], max_l2_norm: float) -> None:
    total_norm = 0
    parameters = list(parameters)
    for p in parameters:
        if p.grad is None:
            continue
        total_norm += torch.sum(p.grad ** 2)

    total_norm = math.sqrt(total_norm)
    if total_norm > max_l2_norm:
        for p in parameters:
            if p.grad is not None:
                p.grad.mul_(max_l2_norm / total_norm + 1e-6)
