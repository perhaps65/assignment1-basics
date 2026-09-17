"""3.4.4–3.4.5 缩放点积注意力与多头自注意力。仅接口骨架，待自行实现。"""

from __future__ import annotations

from torch import Tensor, nn
import torch, math
from cs336_basics.ch03_transformer.s06_softmax import softmax
from cs336_basics.ch03_transformer.s01_linear import Linear
from cs336_basics.ch03_transformer.s05_rope import RotaryPositionalEmbedding


def scaled_dot_product_attention(
    q: Tensor, k: Tensor, v: Tensor, mask: Tensor | None = None,
) -> Tensor:
    d_k = q.shape[-1]
    # 1. QK^T
    attention = torch.einsum("...qd,...kd -> ...qk", q, k)
    # 2. 除 sqrt(d_k)
    attention = attention / math.sqrt(d_k)
    # 3. 如果 mask 不为空，填 -inf
    if mask is not None:
        attention = attention.masked_fill(~mask, -math.inf)
    # 4. softmax
    attention = softmax(attention, dim=-1)
    # 5. attention weights @ V
    attention = attention @ v
    # return
    return attention

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, device=None, dtype=None) -> None:
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_proj = Linear(
            in_features=d_model,
            out_features=d_model,
            device=device,
            dtype=dtype
        )
        self.k_proj = Linear(
            in_features=d_model,
            out_features=d_model,
            device=device,
            dtype=dtype
        )
        self.v_proj = Linear(
            in_features=d_model,
            out_features=d_model,
            device=device,
            dtype=dtype
        )
        self.output_proj = Linear(
            in_features=d_model,
            out_features=d_model,
            device=device,
            dtype=dtype
        )

    def forward(self, x: Tensor) -> Tensor:
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        seq_len = x.shape[-2]
        q = q.reshape(*q.shape[:-1], self.num_heads, self.head_dim).transpose(-2, -3)
        k = k.reshape(*k.shape[:-1], self.num_heads, self.head_dim).transpose(-2, -3)
        v = v.reshape(*v.shape[:-1], self.num_heads, self.head_dim).transpose(-2, -3)
        mask = torch.tril(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device)
        )
        attention = scaled_dot_product_attention(q,k,v,mask).transpose(-2, -3)
        # B T H D_h
        attention = attention.flatten(-2)
        # B T D
        attention = self.output_proj(attention)
        return attention


class MultiHeadSelfAttentionWithRoPE(nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int, theta: float, device=None, dtype=None) -> None:
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.rope = RotaryPositionalEmbedding(
            theta=theta,
            d_k=self.head_dim,
            max_seq_len=max_seq_len,
            device=device
        )
        self.q_proj = Linear(
            in_features=d_model,
            out_features=d_model,
            device=device,
            dtype=dtype
        )
        self.k_proj = Linear(
            in_features=d_model,
            out_features=d_model,
            device=device,
            dtype=dtype
        )
        self.v_proj = Linear(
            in_features=d_model,
            out_features=d_model,
            device=device,
            dtype=dtype
        )
        self.output_proj = Linear(
            in_features=d_model,
            out_features=d_model,
            device=device,
            dtype=dtype
        )

    def forward(self, x: Tensor, token_positions: Tensor | None = None) -> Tensor:
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
    
        seq_len = x.shape[-2]
        q = q.reshape(*q.shape[:-1], self.num_heads, self.head_dim).transpose(-2, -3)
        k = k.reshape(*k.shape[:-1], self.num_heads, self.head_dim).transpose(-2, -3)
        v = v.reshape(*v.shape[:-1], self.num_heads, self.head_dim).transpose(-2, -3)

    
        q = self.rope(q, token_positions)
        k = self.rope(k, token_positions)
        
        mask = torch.tril(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device)
        )
        attention = scaled_dot_product_attention(q,k,v,mask).transpose(-2, -3)
        # B T H D_h
        attention = attention.flatten(-2)
        # B T D
        attention = self.output_proj(attention)
        return attention
