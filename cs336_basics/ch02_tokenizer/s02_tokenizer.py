"""2.6 BPE 编码与解码。仅接口骨架，待自行实现。"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from os import PathLike
import regex as re
import json

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class Tokenizer:
    def __init__(
        self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        self.merges = merges
        self.vocab = dict(vocab)
        # vocab中没有特殊token的话,要补充上
        special_tokens = [] if special_tokens is None else special_tokens
        next_index = len(vocab)
        for special_token in special_tokens:
            encode = special_token.encode("utf-8")
            if encode in self.vocab.values():
                continue
            self.vocab[next_index] = encode
            next_index += 1
        self.special_tokens = special_tokens
        self.reverse_vocab = {v:k for k, v in self.vocab.items()}
        self.merge_rank = {
                    pair: i
                    for i, pair in enumerate(self.merges)
        }

    @classmethod
    def from_files(
        cls, vocab_filepath: str | PathLike[str], merges_filepath: str | PathLike[str],
        special_tokens: list[str] | None = None,
    ) -> Tokenizer:
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            raw_vocab: dict[str, int] = json.load(f)
            raw_vocab = {
                v:k.encode("utf-8")
                for k, v in raw_vocab.items()
            }
        merges = []
        with open(merges_filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                if not line:
                    continue

                split = line.split()
                left = split[0].encode("utf-8")
                right = split[1].encode("utf-8")
                merges.append((left, right))
        return cls(
            raw_vocab, merges, special_tokens
        )


    def encode(self, text: str) -> list[int]:
        if len(self.special_tokens) > 0:
            escaped_tokens = []
            for special_token in sorted(self.special_tokens, key=len, reverse=True):
                escaped_tokens.append(re.escape(special_token))
            regex_text = "(" + "|".join(escaped_tokens) + ")"
            split_texts = re.split(regex_text, text)
        else:
            split_texts = [text]
        ids: list[int] = []
        # 排名dict
        
        for split_text in split_texts:
            if split_text == "":
                continue
            if split_text in self.special_tokens:
                ids.append(self.reverse_vocab[split_text.encode("utf-8")])
                continue
            # 预分词
            for match in re.finditer(PAT, split_text):
                raw_bytes = match.group().encode("utf-8")
                bytes_list = [
                    raw_bytes[i:i + 1]
                    for i in range(len(raw_bytes))
                ]
                # merge
                while True:
                    best_pair = None
                    low_rank = 0x7fffffff
                    new_bytes_list = []
                    for i in range(len(bytes_list) - 1):
                        pair = (bytes_list[i], bytes_list[i + 1])
                        if pair in self.merge_rank and self.merge_rank[pair] < low_rank:
                            best_pair = pair
                            low_rank = self.merge_rank[pair]
                    if best_pair is None:
                        break
                    i = 0
                    length = len(bytes_list)
                    while i < length:
                        if i < length - 1 and (bytes_list[i], bytes_list[i + 1]) == best_pair:
                            new_bytes_list.append(bytes_list[i] + bytes_list[i + 1])
                            i += 2
                        else:
                            new_bytes_list.append(bytes_list[i])
                            i += 1
                    bytes_list = new_bytes_list
                for bytes_segment in bytes_list:
                    ids.append(self.reverse_vocab[bytes_segment])
        return ids        
                

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        # 此处只decode一次 不能单独bytes解码
        decode_bytes = [self.vocab[id] for id in ids]
        return b"".join(decode_bytes).decode("utf-8", errors="replace")
