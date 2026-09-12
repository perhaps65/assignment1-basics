"""2.6 BPE 编码与解码。仅接口骨架，待自行实现。"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from os import PathLike


class Tokenizer:
    def __init__(
        self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        raise NotImplementedError

    @classmethod
    def from_files(
        cls, vocab_filepath: str | PathLike[str], merges_filepath: str | PathLike[str],
        special_tokens: list[str] | None = None,
    ) -> Tokenizer:
        raise NotImplementedError

    def encode(self, text: str) -> list[int]:
        raise NotImplementedError

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        raise NotImplementedError

    def decode(self, ids: list[int]) -> str:
        raise NotImplementedError
