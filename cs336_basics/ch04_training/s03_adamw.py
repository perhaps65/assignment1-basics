"""4.3 AdamW。仅接口骨架，待自行实现。"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from torch import Tensor, nn
from torch.optim import Optimizer
import math,torch


class AdamW(Optimizer):
    def __init__(
        self, params: Iterable[nn.Parameter] | Iterable[dict[str, Any]],
        lr: float = 1e-3, betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8, weight_decay: float = 0.01,
    ) -> None:
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay
        }
        super().__init__(params, defaults)

    def step(self, closure: Callable[[], Tensor] | None = None) -> Tensor | None:
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1 = group["betas"][0]
            beta2 = group["betas"][1]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            
            for p in group["params"]:
                grad = p.grad
                if grad is None:
                    continue
                # 注意是当前p的state
                state = self.state[p]
                if len(state) == 0:
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                    state["t"] = 0
                state["t"] += 1

                alpha_lr = lr * math.sqrt(1 - beta2**state["t"]) / (1 - beta1**state["t"])
                state["m"] = beta1 * state["m"] + (1 - beta1) * grad
                state["v"] = beta2 * state["v"] + (1 - beta2) * (grad**2)
                with torch.no_grad():
                    # weight decay
                    p.data -= p.data * lr * weight_decay
                    
                    # Adam方式
                    p.data -= alpha_lr * state["m"] / (torch.sqrt(state["v"]) + eps)
                
        return loss        

