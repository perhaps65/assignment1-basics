"""4.3 AdamW。仅接口骨架，待自行实现。"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from torch import Tensor, nn
from torch.optim import Optimizer


class AdamW(Optimizer):
    def __init__(
        self, params: Iterable[nn.Parameter] | Iterable[dict[str, Any]],
        lr: float = 1e-3, betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8, weight_decay: float = 0.01,
    ) -> None:
        raise NotImplementedError

    def step(self, closure: Callable[[], Tensor] | None = None) -> Tensor | None:
        raise NotImplementedError
