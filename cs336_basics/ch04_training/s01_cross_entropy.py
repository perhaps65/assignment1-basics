"""4.1 交叉熵损失。仅接口骨架，待自行实现。"""

from __future__ import annotations

from torch import Tensor
import torch


def cross_entropy(inputs: Tensor, targets: Tensor) -> Tensor:
    # -log(exp(correct - max) / sum(exp(logits - max)))
    # log(sum(exp(logits - max))) - correct + max
    # [B,T] - [B,T]
    correct = torch.gather(inputs, dim=-1, index=targets.unsqueeze(-1)).squeeze(dim=-1) 
    # [B,T,1]->[B,T]
    max_value = inputs.max(dim=-1,keepdim=True).values
    log_sum_exp = torch.log(torch.sum(torch.exp(inputs - max_value), dim=-1, keepdim=False))
    loss = log_sum_exp - correct + max_value.squeeze(dim=-1)
    return loss.mean()
