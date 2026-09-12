# 作业文件索引

按作业 PDF 章节组织。章节目录和模块的数字前缀用于 IDE 按名称排序；文件名和类名是项目自选约定，可自行调整。

所有函数和方法仅有签名与 NotImplementedError 占位；main 脚本执行时也会提示尚未实现。import 当前只包含签名及基类所需依赖，算法依赖请在实现时添加。

| 章节 | 文件 | 题目 / 用途 |
| --- | --- | --- |
| 2.4–2.5 | [ch02_tokenizer/s01_train_bpe.py](ch02_tokenizer/s01_train_bpe.py) | BPE 训练与训练实验 |
| 2.6 | [ch02_tokenizer/s02_tokenizer.py](ch02_tokenizer/s02_tokenizer.py) | BPE 编码与解码 |
| 2.7 | [ch02_tokenizer/s03_tokenizer_experiments.py](ch02_tokenizer/s03_tokenizer_experiments.py) | 分词器实验 |
| 3.3.2 | [ch03_transformer/s01_linear.py](ch03_transformer/s01_linear.py) | Linear 模块 |
| 3.3.3 | [ch03_transformer/s02_embedding.py](ch03_transformer/s02_embedding.py) | Embedding 模块 |
| 3.4.1 | [ch03_transformer/s03_rmsnorm.py](ch03_transformer/s03_rmsnorm.py) | RMSNorm |
| 3.4.2 | [ch03_transformer/s04_feedforward.py](ch03_transformer/s04_feedforward.py) | SiLU 与 SwiGLU |
| 3.4.3 | [ch03_transformer/s05_rope.py](ch03_transformer/s05_rope.py) | 旋转位置编码 |
| 3.4.4 | [ch03_transformer/s06_softmax.py](ch03_transformer/s06_softmax.py) | Softmax |
| 3.4.4–3.4.5 | [ch03_transformer/s07_attention.py](ch03_transformer/s07_attention.py) | 缩放点积注意力与多头自注意力 |
| 3.4–3.5 | [ch03_transformer/s08_transformer_block.py](ch03_transformer/s08_transformer_block.py) | Transformer Block |
| 3.5 | [ch03_transformer/s09_transformer_lm.py](ch03_transformer/s09_transformer_lm.py) | 完整 Transformer LM |
| 4.1 | [ch04_training/s01_cross_entropy.py](ch04_training/s01_cross_entropy.py) | 交叉熵损失 |
| 4.2 | [ch04_training/s02_sgd_experiments.py](ch04_training/s02_sgd_experiments.py) | SGD 实验 |
| 4.3 | [ch04_training/s03_adamw.py](ch04_training/s03_adamw.py) | AdamW |
| 4.4 | [ch04_training/s04_lr_schedule.py](ch04_training/s04_lr_schedule.py) | 学习率调度 |
| 4.5 | [ch04_training/s05_gradient_clipping.py](ch04_training/s05_gradient_clipping.py) | 梯度裁剪 |
| 5.1 | [ch05_training_loop/s01_data.py](ch05_training_loop/s01_data.py) | 数据加载 |
| 5.2 | [ch05_training_loop/s02_checkpoint.py](ch05_training_loop/s02_checkpoint.py) | 检查点保存与恢复 |
| 5.3 | [ch05_training_loop/s03_train.py](ch05_training_loop/s03_train.py) | 训练主程序 |
| 6 | [ch06_generation/s01_generate.py](ch06_generation/s01_generate.py) | 文本生成主程序 |
| 7 | [ch07_experiments/s01_experiments.py](ch07_experiments/s01_experiments.py) | 训练实验与消融实验 |

各章节目录含 __init__.py，可作为 Python 子包导入。测试适配器仍位于 tests/adapters.py；完成实现后自行连接上述模块。

例如，BPE 训练模块的导入路径为 `cs336_basics.ch02_tokenizer.s01_train_bpe`；训练入口可通过 `python -m cs336_basics.ch05_training_loop.s03_train` 调用（目前会抛出 NotImplementedError）。

第 1 章为作业说明；概念问答、资源核算及实验结果请记录在单独的作业报告中。原有根目录 __init__.py、pretokenization_example.py 和测试文件未修改。
