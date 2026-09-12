"""2.4–2.5 BPE 训练与训练实验。仅接口骨架，待自行实现。"""

from __future__ import annotations

from os import PathLike


def train_bpe(
    input_path: str | PathLike[str],
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    raise NotImplementedError
