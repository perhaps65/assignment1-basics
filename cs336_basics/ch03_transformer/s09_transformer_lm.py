"""3.5 完整 Transformer LM。仅接口骨架，待自行实现。"""

from __future__ import annotations

from torch import Tensor, nn
from cs336_basics.ch03_transformer.s08_transformer_block import TransformerBlock
from cs336_basics.ch03_transformer.s03_rmsnorm import RMSNorm
from cs336_basics.ch03_transformer.s02_embedding import Embedding
from cs336_basics.ch03_transformer.s01_linear import Linear

class TransformerLM(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, d_model: int, num_layers: int, num_heads: int, d_ff: int, rope_theta: float, device=None, dtype=None) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                max_seq_len=context_length,
                theta=rope_theta,
                device=device,
                dtype=dtype
            )
            for _ in range(num_layers)
        )
        self.ln_final = RMSNorm(
            d_model=d_model,
            device=device,
            dtype=dtype
        )
        self.token_embeddings = Embedding(
            embedding_dim=d_model,
            num_embeddings=vocab_size,
            device=device,
            dtype=dtype
        )
        self.lm_head = Linear(
            in_features=d_model,
            out_features=vocab_size,
            device=device,
            dtype=dtype
        )

    def forward(self, token_ids: Tensor) -> Tensor:
        x = self.token_embeddings(token_ids)
        for attn_layer in self.layers:
            x = attn_layer(x)
        x = self.ln_final(x)
        logits = self.lm_head(x)
        return logits
