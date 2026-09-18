"""5.3 TinyStories 训练主程序：配置与接口骨架，核心逻辑待自行实现。

查看参数：
    python -m cs336_basics.ch05_training_loop.s03_train --help

数据路径应指向已经编码的 token 文件，而不是原始文本。
默认参数用于检查流程，不代表正式训练配置。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy.typing as npt
from torch import nn


@dataclass
class TrainConfig:
    # 必填：与自己的 tokenizer 和预处理结果一致。
    train_data_path: Path
    val_data_path: Path
    vocab_size: int

    # 数据：明确区分 .npy 文件和无文件头的原始二进制文件。
    data_format: str = "npy"
    token_dtype: str = "uint16"  # 仅用于 raw；必须与写入时的 dtype 一致。

    # 模型：字段名对应 TransformerLM.__init__。
    context_length: int = 128
    d_model: int = 128
    num_layers: int = 2
    num_heads: int = 4
    d_ff: int = 344
    rope_theta: float = 10000.0

    # 运行环境：根据实际 PyTorch 环境选择设备。
    device: str = "cuda"
    dtype: str = "float32"
    seed: int = 42

    # 优化器与训练预算。
    batch_size: int = 8
    max_steps: int = 200
    max_lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 20
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0

    # 日志、验证与恢复。
    log_interval: int = 10
    eval_interval: int = 50
    eval_batches: int = 10
    save_interval: int = 100
    checkpoint_dir: Path = Path("checkpoints/tinystories")
    resume_from: Path | None = None


def parse_args() -> TrainConfig:
    """常用参数支持命令行覆盖，其余参数先在 TrainConfig 中调整。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data-path", type=Path, required=True)
    parser.add_argument("--val-data-path", type=Path, required=True)
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--data-format", choices=("npy", "raw"), default=TrainConfig.data_format)
    parser.add_argument("--token-dtype", choices=("uint16", "uint32", "int64"), default=TrainConfig.token_dtype)
    parser.add_argument("--device", default=TrainConfig.device)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--context-length", type=int, default=TrainConfig.context_length)
    parser.add_argument("--max-steps", type=int, default=TrainConfig.max_steps)
    parser.add_argument("--warmup-steps", type=int, default=TrainConfig.warmup_steps)
    parser.add_argument("--max-lr", type=float, default=TrainConfig.max_lr)
    parser.add_argument("--min-lr", type=float, default=TrainConfig.min_lr)
    parser.add_argument("--checkpoint-dir", type=Path, default=TrainConfig.checkpoint_dir)
    parser.add_argument("--resume-from", type=Path)
    return TrainConfig(**vars(parser.parse_args()))


def load_datasets(config: TrainConfig) -> tuple[npt.NDArray, npt.NDArray]:
    """返回训练与验证的一维 token 数组，供 s01_data.get_batch 使用。

    TODO：实现数据读取。检查文件格式、整数类型、token 范围及数据长度。
    """
    raise NotImplementedError("待实现：读取已编码的训练集与验证集")


def evaluate(model: nn.Module, val_data: npt.NDArray, config: TrainConfig) -> float:
    """返回验证集的平均 token loss。

    TODO：实现验证。注意梯度开关、模型模式恢复及 loss 的统计口径。
    """
    raise NotImplementedError("待实现：验证损失统计")


def train(config: TrainConfig) -> None:
    """实现训练管理，复用已经完成的组件。

    可用接口：
    - ch03_transformer.s09_transformer_lm.TransformerLM
    - ch04_training 中的交叉熵、AdamW、学习率调度和梯度裁剪
    - s01_data.get_batch
    - s02_checkpoint.save_checkpoint / load_checkpoint

    TODO：设备与随机种子、模型与优化器初始化、训练、日志和保存恢复。
    检查点的 iteration 含义需统一，例如约定为已完成的更新次数。
    """
    raise NotImplementedError("待实现：训练主流程")


def main() -> None:
    config = parse_args()
    train(config)


if __name__ == "__main__":
    main()
