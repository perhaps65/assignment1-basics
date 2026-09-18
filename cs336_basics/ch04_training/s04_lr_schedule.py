"""4.4 学习率调度。仅接口骨架，待自行实现。"""

from __future__ import annotations
import math

def get_lr_cosine_schedule(
    it: int, max_learning_rate: float, min_learning_rate: float,
    warmup_iters: int, cosine_cycle_iters: int,
) -> float:
    if it <= warmup_iters:
        return it * max_learning_rate / warmup_iters
    if it < cosine_cycle_iters:
        progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        return min_learning_rate + 0.5 * (1 + math.cos(math.pi * progress)) * (max_learning_rate - min_learning_rate)
    return min_learning_rate
