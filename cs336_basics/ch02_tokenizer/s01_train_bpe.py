"""2.4–2.5 BPE 训练与训练实验。仅接口骨架，待自行实现。"""

from __future__ import annotations

from os import PathLike
import regex as re
from collections import defaultdict

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def train_bpe(
    input_path: str | PathLike[str],
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
    # 分词表
    vocab = {}
    merge_list = []
    for idx in range(0, 256):
        vocab[idx] = bytes([idx])
    # 分隔符划分
    next_id = 256
    if len(special_tokens) > 0:
        escaped_token = []
        for token in special_tokens:
            escaped_token.append(re.escape(token))
            vocab[next_id] = token.encode("utf-8")
            next_id += 1
        pattern = "|".join(escaped_token)
        text_list = re.split(pattern, text)
    else:
        text_list = [text]
    # 预分词
    pre_tokens = defaultdict(int)
    for split_text in text_list:
        for match in re.finditer(PAT, split_text):
            encode = match.group().encode("utf-8")
            pre_token = tuple(
                bytes([num])
                for num in encode
            )
            pre_tokens[pre_token] += 1
    # 循环
    while next_id < vocab_size:
        pair_counts = defaultdict(int)
        for pre_token, freq in pre_tokens.items():
            for i in range(len(pre_token) - 1):
                pair_counts[(pre_token[i], pre_token[i + 1])] += freq
        # 查找字节最大出现次数pair
        if len(pair_counts) == 0:
            break
        left, right = max(pair_counts, key=lambda pair : (pair_counts[pair], pair))
        new_token = left + right
        new_pre_tokens = defaultdict(int)
        # 更新token表
        for pre_token, freq in pre_tokens.items():
            new_token_list = []
            token_len = len(pre_token)
            i = 0
            while i < token_len:
                if i + 1 < token_len and pre_token[i] == left and pre_token[i + 1] == right:
                    new_token_list.append(new_token)
                    i += 2
                else:
                    new_token_list.append(pre_token[i])
                    i += 1
            new_pre_tokens[tuple(new_token_list)] += freq
        pre_tokens = new_pre_tokens
        # 更新分词表
        vocab[next_id] = new_token
        next_id += 1
        # 记录合并
        merge_list.append((left, right))
    return vocab, merge_list
