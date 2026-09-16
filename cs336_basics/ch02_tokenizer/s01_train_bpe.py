"""2.4–2.5 BPE 训练与训练实验。仅接口骨架，待自行实现。"""

from __future__ import annotations

from os import PathLike
import regex as re
from collections import Counter, defaultdict
from collections import deque
from pathlib import Path
import sys
import time
import heapq

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class _MaxPair(tuple):
    """让最小堆按 (频次, 字节 pair) 的降序取元素，保留原来的平局规则。"""

    def __lt__(self, other):
        return tuple.__gt__(self, other)


def train_bpe(
    input_path: str | PathLike[str],
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """训练 BPE；verbose=False 关闭日志，log_interval 控制日志间隔（秒）。"""
    verbose = kwargs.get("verbose", True)
    log_interval = float(kwargs.get("log_interval", 10.0))
    started = time.perf_counter()
    last_log = started

    def duration(seconds):
        if seconds is None:
            return "估算中"
        seconds = max(0, int(seconds))
        return f"{seconds // 3600:02d}:{seconds // 60 % 60:02d}:{seconds % 60:02d}"

    def log(message, force=False):
        nonlocal last_log
        now = time.perf_counter()
        if verbose and (force or now - last_log >= log_interval):
            print(f"[BPE 耗时 {duration(now - started)}] {message}", file=sys.stderr, flush=True)
            last_log = now

    file_size = Path(input_path).stat().st_size
    log(f"开始训练：文件={input_path}，大小={file_size / 1024**2:.1f} MiB，目标词表={vocab_size:,}", True)
    # with open(input_path, "r", encoding="utf-8") as f:
    #     text = f.read()
    # 分词表
    vocab = {}
    merge_list = []
    for idx in range(0, 256):
        vocab[idx] = bytes([idx])
    # 分隔符划分
    next_id = 256
    if len(special_tokens) > 0:
        for token in special_tokens:
            vocab[next_id] = token.encode("utf-8")
            next_id += 1
    # 预分词
    pre_tokens = defaultdict(int)
    documents = 0
    processed_bytes = 0
    token_count = 0
    log("预分词开始（此阶段 ETA 不包含后续合并）", True)
    for split_text in iter_documents(
        input_path,
        delimiter="<|endoftext|>",
    ):
        for match in re.finditer(PAT, split_text):
            encode = match.group().encode("utf-8")

            pre_token = tuple(
                bytes([num])
                for num in encode
            )

            pre_tokens[pre_token] += 1
            token_count += 1
            if verbose and token_count % 100_000 == 0:
                log(f"预分词：已完成文档={documents:,}，预分词数={token_count:,}，不同词={len(pre_tokens):,}；正在处理文档")
        documents += 1
        processed_bytes = min(file_size, processed_bytes + len(split_text.encode("utf-8")) + len(b"<|endoftext|>"))
        elapsed = time.perf_counter() - started
        eta = elapsed * (file_size - processed_bytes) / processed_bytes if processed_bytes else None
        log(f"预分词：{processed_bytes / max(file_size, 1):.1%}，文档={documents:,}，不同词={len(pre_tokens):,}，阶段 ETA≈{duration(eta)}")
    log(f"预分词完成：文档={documents:,}，预分词数={token_count:,}，不同词={len(pre_tokens):,}", True)
    merge_target = max(0, vocab_size - next_id)
    recent_merge_times = deque(maxlen=20)
    # 为每个不同的预分词保留稳定 ID 和语料频次。
    words = list(pre_tokens)
    frequencies = list(pre_tokens.values())
    del pre_tokens
    pair_counts = defaultdict(int)
    pair_words = defaultdict(set)
    log("初始化 pair 计数和倒排索引", True)
    for word_id, word in enumerate(words):
        for pair, count in Counter(zip(word, word[1:])).items():
            pair_counts[pair] += count * frequencies[word_id]
            pair_words[pair].add(word_id)
        if verbose and (word_id + 1) % 10_000 == 0:
            log(f"初始化索引：{word_id + 1:,}/{len(words):,}")
    heap = [_MaxPair((count, pair)) for pair, count in pair_counts.items()]
    heapq.heapify(heap)
    log(f"合并开始：计划最多 {merge_target:,} 轮；ETA 按最近 20 轮平均耗时估算", True)
    # 循环
    while next_id < vocab_size:
        round_started = time.perf_counter()
        round_number = len(merge_list) + 1
        # 丢弃历史频次；每轮只为计数发生变化的 pair 添加新的堆项。
        while heap:
            count, pair = heapq.heappop(heap)
            if pair_counts.get(pair, 0) == count:
                break
        else:
            log("无可合并的 pair，提前结束", True)
            break
        left, right = pair
        new_token = left + right
        affected_words = tuple(pair_words[pair])
        word_count = len(affected_words)
        changed_pairs = set()
        selected_at = time.perf_counter()
        for word_index, word_id in enumerate(affected_words, 1):
            pre_token = words[word_id]
            freq = frequencies[word_id]
            old_pairs = Counter(zip(pre_token, pre_token[1:]))
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
            updated_word = tuple(new_token_list)
            words[word_id] = updated_word
            new_pairs = Counter(zip(updated_word, updated_word[1:]))
            # 未变化的相邻 pair 不动；重复 pair 按出现次数及语料频次加权。
            # 比较合并前后计数，也覆盖 aaa/aaaa 中的重叠相邻 pair。
            for adjacent_pair in old_pairs.keys() | new_pairs.keys():
                delta = new_pairs[adjacent_pair] - old_pairs[adjacent_pair]
                if delta:
                    pair_counts[adjacent_pair] += delta * freq
                    changed_pairs.add(adjacent_pair)
                if adjacent_pair not in new_pairs:
                    pair_words[adjacent_pair].discard(word_id)
                elif adjacent_pair not in old_pairs:
                    pair_words[adjacent_pair].add(word_id)
            if verbose and word_index % 10_000 == 0:
                log(f"合并 {round_number:,}/{merge_target:,}：更新受影响词 {word_index:,}/{word_count:,} ({word_index / word_count:.1%})")
        for changed_pair in changed_pairs:
            count = pair_counts[changed_pair]
            if count:
                heapq.heappush(heap, _MaxPair((count, changed_pair)))
            else:
                del pair_counts[changed_pair]
                del pair_words[changed_pair]
        # 限制惰性删除留下的旧堆项，避免长时间训练时无限积累。
        if len(heap) > max(1024, 3 * len(pair_counts)):
            heap = [_MaxPair((count, pair)) for pair, count in pair_counts.items()]
            heapq.heapify(heap)
        # 更新分词表
        vocab[next_id] = new_token
        next_id += 1
        # 记录合并
        merge_list.append((left, right))
        now = time.perf_counter()
        recent_merge_times.append(now - round_started)
        eta = sum(recent_merge_times) / len(recent_merge_times) * (merge_target - len(merge_list))
        log(
            f"合并完成 {len(merge_list):,}/{merge_target:,} ({len(merge_list) / merge_target:.1%})，"
            f"词表={next_id:,}，本轮={now - round_started:.2f}s "
            f"(选择={selected_at - round_started:.2f}s，增量更新={now - selected_at:.2f}s，受影响词={word_count:,})，"
            f"合并阶段 ETA≈{duration(eta)}",
            force=len(merge_list) == 1 or now - round_started >= log_interval,
        )
    log(f"训练完成：词表={len(vocab):,}，合并={len(merge_list):,}", True)
    return vocab, merge_list


def iter_documents(
    input_path: str | PathLike[str],
    delimiter: str = "<|endoftext|>",
    chunk_size: int = 8 * 1024 * 1024,
):
    buffer = ""

    with open(
        input_path,
        "r",
        encoding="utf-8",
    ) as f:
        while True:
            chunk = f.read(chunk_size)

            if not chunk:
                break

            buffer += chunk

            parts = buffer.split(delimiter)

            # 最后这一段可能只是半篇文档，
            # 留给下一块继续拼
            buffer = parts.pop()

            for part in parts:
                yield part

        # 文件末尾没有 delimiter 的残余
        if buffer:
            yield buffer
