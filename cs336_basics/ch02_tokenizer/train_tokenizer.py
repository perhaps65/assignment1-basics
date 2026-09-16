from __future__ import annotations

import pickle
import time
from pathlib import Path

from datasets import load_dataset

from cs336_basics.ch02_tokenizer.s01_train_bpe import train_bpe


SPECIAL_TOKENS = ["<|endoftext|>"]


def export_owt_sample() -> Path:
    ds = load_dataset(
        "stanford-cs336/owt-sample",
        split="validation",
    )

    output_path = Path("/home/administrator/projects/cs336/assignment1-basics/tests/fixtures/owt_sample_train.txt")
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:
        for example in ds:
            f.write(example["text"])
            f.write("<|endoftext|>")

    print(
        f"exported {len(ds)} documents "
        f"to {output_path}"
    )

    return output_path


def train_and_save(
    input_path: str | Path,
    vocab_size: int,
    output_path: str | Path,
) -> None:

    start = time.perf_counter()

    vocab, merges = train_bpe(
        input_path=input_path,
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
    )

    elapsed = time.perf_counter() - start

    longest_token = max(
        vocab.values(),
        key=len,
    )

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(output_path, "wb") as f:
        pickle.dump(
            {
                "vocab": vocab,
                "merges": merges,
            },
            f,
        )

    print("vocab size:", len(vocab))
    print("merges:", len(merges))
    print(f"time: {elapsed:.2f} s")

    print(
        "longest token:",
        longest_token,
    )

    print(
        "longest token decoded:",
        longest_token.decode(
            "utf-8",
            errors="replace",
        ),
    )

    print(
        "longest token bytes:",
        len(longest_token),
    )

    print(
        "saved to:",
        output_path,
    )


def main() -> None:
    # owt_path = export_owt_sample()

    train_and_save(
        input_path="/home/administrator/projects/cs336/assignment1-basics/tests/fixtures/owt_train.txt",
        vocab_size=32_000,
        output_path="artifacts/owt_32k.pkl",
    )


if __name__ == "__main__":
    main()