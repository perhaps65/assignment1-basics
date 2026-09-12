"""4.4 学习率调度。仅接口骨架，待自行实现。"""

from __future__ import annotations


def get_lr_cosine_schedule(
    it: int, max_learning_rate: float, min_learning_rate: float,
    warmup_iters: int, cosine_cycle_iters: int,
) -> float:
    raise NotImplementedError
