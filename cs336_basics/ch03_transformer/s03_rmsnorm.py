"""3.4.1 RMSNorm。仅接口骨架，待自行实现。"""

from __future__ import annotations

from torch import Tensor, nn
import torch, math

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.ones(
                d_model,
                device=device,
                dtype=dtype
            )
        )
        self.eps = eps


    def forward(self, x: Tensor) -> Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        mean_square = x.pow(2).mean(dim=-1, keepdim=True)#B T 1
        rms = torch.sqrt(mean_square + self.eps)
        result = (x / rms) * self.weight
        return result.to(in_dtype)
