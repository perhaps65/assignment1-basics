"""3.4–3.5 Transformer Block。仅接口骨架，待自行实现。"""

from __future__ import annotations

from torch import Tensor, nn
from cs336_basics.ch03_transformer.s07_attention import MultiHeadSelfAttentionWithRoPE
from cs336_basics.ch03_transformer.s03_rmsnorm import RMSNorm
from cs336_basics.ch03_transformer.s04_feedforward import SwiGLU


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float, device=None, dtype=None) -> None:
        super().__init__()
        self.attn = MultiHeadSelfAttentionWithRoPE(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            theta=theta,
            device=device,
            dtype=dtype
        )
        self.ln1 = RMSNorm(
            d_model=d_model,
            device=device,
            dtype=dtype
        )
        self.ln2 = RMSNorm(
            d_model=d_model,
            device=device,
            dtype=dtype
        )
        self.ffn = SwiGLU(
            d_model=d_model,
            d_ff=d_ff,
            device=device,
            dtype=dtype
        )

    def forward(self, x: Tensor) -> Tensor:
        origin = x
        x = self.ln1(x)
        x = self.attn(x)
        x = x + origin

        origin = x
        x = self.ln2(x)
        x = self.ffn(x)
        x = x + origin

        return x


