"""2.7 Tokenizer experiments."""

from __future__ import annotations

import pickle
import time
from pathlib import Path

from cs336_basics.ch02_tokenizer.s02_tokenizer import Tokenizer

END_OF_TEXT = "<|endoftext|>"

TINYSTORIES_DATA = "/home/administrator/projects/cs336/assignment1-basics/tests/fixtures/tinystories_sample_5M.txt"
OWT_DATA = "/home/administrator/projects/cs336/assignment1-basics/tests/fixtures/owt_train.txt"

TINYSTORIES_TOKENIZER = "artifacts/tinystories_10k.pkl"
OWT_TOKENIZER = "artifacts/owt_32k.pkl"


def load_tokenizer(path: str) -> Tokenizer:
    with open(path, "rb") as f:
        data = pickle.load(f)

    return Tokenizer(
        vocab=data["vocab"],
        merges=data["merges"],
        special_tokens=[END_OF_TEXT],
    )


def load_documents(
    path: str | Path,
    n: int,
) -> list[str]:
    """读取前 n 篇以 <|endoftext|> 分隔的文档，不读取整个大文件。"""

    docs = []
    buffer = ""

    with open(path, "r", encoding="utf-8") as f:
        while len(docs) < n:
            chunk = f.read(8 * 1024 * 1024)

            if not chunk:
                break

            # 1. chunk 拼到 buffer
            buffer += chunk

            # 2. 用 END_OF_TEXT 切开
            splits = buffer.split(END_OF_TEXT)

            # 3. 最后一块可能是不完整文档，留回 buffer
            buffer = splits.pop()

            # 4. 完整文档加入 docs
            docs.extend(doc for doc in splits if doc.strip())

            # 5. 满 n 个停止
            if len(docs) >= n:
                break

    return docs[:n]


def compression_ratio(
    tokenizer: Tokenizer,
    docs: list[str],
) -> float:
    total_bytes = 0
    total_tokens = 0

    for doc in docs:
        total_bytes += len(doc.encode("utf-8"))
        total_tokens += len(tokenizer.encode(doc))

    return total_bytes / total_tokens


def benchmark_tokenizer(
    tokenizer: Tokenizer,
    text: str,
) -> float:
    num_bytes = len(text.encode("utf-8"))

    start = time.perf_counter()

    tokenizer.encode(text)

    elapsed = time.perf_counter() - start

    return num_bytes / elapsed


def main() -> None:
    tiny_tokenizer = load_tokenizer(TINYSTORIES_TOKENIZER)
    owt_tokenizer = load_tokenizer(OWT_TOKENIZER)

    tiny_docs = load_documents(TINYSTORIES_DATA, 10)
    owt_docs = load_documents(OWT_DATA, 10)

    # -------------------------
    # (a)
    # -------------------------

    tiny_ratio = compression_ratio(tiny_tokenizer, tiny_docs)
    owt_ratio = compression_ratio(owt_tokenizer, owt_docs)

    print("(a)")
    print("TinyStories tokenizer on TinyStories:", tiny_ratio, "bytes/token")
    print("OWT tokenizer on OWT:", owt_ratio, "bytes/token")

    # -------------------------
    # (b)
    # -------------------------

    tiny_on_owt_ratio = compression_ratio(tiny_tokenizer, owt_docs)

    print("\n(b)")
    print("TinyStories tokenizer on OWT:", tiny_on_owt_ratio, "bytes/token")
    print("OWT tokenizer on OWT:", owt_ratio, "bytes/token")

    # -------------------------
    # (c)
    # -------------------------

    benchmark_text = "".join(owt_docs)

    throughput = benchmark_tokenizer(owt_tokenizer, benchmark_text)

    pile_bytes = 825e9
    pile_seconds = pile_bytes / throughput
    pile_hours = pile_seconds / 3600

    print("\n(c)")
    print("throughput:", throughput, "bytes/s")
    print("Pile estimate:", pile_hours, "hours")


if __name__ == "__main__":
    main()