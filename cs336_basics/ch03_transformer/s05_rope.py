"""3.4.3 旋转位置编码。仅接口骨架，待自行实现。"""

from __future__ import annotations

from torch import Tensor, nn
import torch


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None) -> None:
        super().__init__()
        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        dims = torch.arange(0, d_k, 2, device=device, dtype=torch.float32,)
        freq = theta ** (-dims / d_k)
        positions = positions[:,None]
        freq = freq[None,:]
        angle = freq * positions
        self.register_buffer("cos", torch.cos(angle), persistent=False)
        self.register_buffer("sin", torch.sin(angle), persistent=False)

    def forward(self, x: Tensor, token_positions: Tensor) -> Tensor:
        # b h t d_h
        # b t 
        even = x[..., 0::2]
        odd = x[..., 1::2]
        # b h t d_h/2
        if token_positions is None:
            # 无论是不是多头 [-2]都是T,后面计算cos sin都可以广播,
            token_positions = torch.arange(x.shape[-2]) 
        else:
            token_positions = token_positions.unsqueeze(-2)
        cos = self.cos[token_positions]
        sin = self.sin[token_positions]
        # b t d_h/2
        rotated_even = even * cos - odd * sin
        rotated_odd = even * sin + odd * cos
        rotate = torch.stack([rotated_even, rotated_odd], dim=-1).flatten(-2)
        
        return rotate

