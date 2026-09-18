"""5.1 数据加载。仅接口骨架，待自行实现。"""

from __future__ import annotations

import numpy.typing as npt
import numpy as np
import torch
from torch import Tensor


def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str,
) -> tuple[Tensor, Tensor]:
    startArray = np.random.randint(0, len(dataset) - context_length, size=batch_size)
    inputs = []
    outputs = []
    for t in startArray:
        inputs.append(torch.tensor(dataset[t:t + context_length]))
        outputs.append(torch.tensor(dataset[t + 1:t + context_length + 1]))
    input_tensor = torch.stack(inputs, dim=0).to(dtype=torch.long, device=device)
    output_tensor = torch.stack(outputs, dim=0).to(dtype=torch.long, device=device)
    return (input_tensor, output_tensor)


