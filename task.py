import itertools
from typing import List, Tuple

import chz
import torch

K = 4  # bits per task subset, fixed


@chz.chz
class ParityTaskConfig:
    n: int = 64          # bit-string length
    n_tasks: int = 4     # number of atomic tasks
    codes: str = "[[0], [1], [2], [3], [0, 1, 2, 3]]"


def get_subsets(n_tasks: int) -> List[List[int]]:
    return [[i * K + j for j in range(K)] for i in range(n_tasks)]


def get_batch(
    n_tasks: int,
    n: int,
    subsets: List[List[int]],
    task_codes: List[List[int]],
    batch_sizes: List[int],
    dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device("cpu"),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create a batch for sparse parity tasks.

    Parameters
    ----------
    n_tasks : int
        Number of atomic tasks (must equal len(subsets)).
    n : int
        Length of each random bit-string.
    subsets : List[List[int]]
        Each subsets[i] is a list of zero-based bit-positions in [0..n-1].
    task_codes : List[List[int]]
        Which atomic tasks to combine for each sample.
    batch_sizes : List[int]
        Number of samples per code; same length as task_codes.
    dtype : torch.dtype
        Dtype for `x`. Output `y` is torch.int64.
    device : torch.device
        Torch device.

    Returns
    -------
    x : torch.Tensor, shape (sum(batch_sizes), n_tasks + n)
    y : torch.Tensor, shape (sum(batch_sizes),)
    """
    assert len(subsets) == n_tasks, "Need exactly one subset per atomic task"
    assert len(task_codes) == len(batch_sizes)

    total = sum(batch_sizes)
    x = torch.zeros((total, n_tasks + n), dtype=dtype, device=device)
    bits = torch.randint(0, 2, (total, n), dtype=dtype, device=device)
    x[:, n_tasks:] = bits

    y = torch.empty((total,), dtype=torch.int64, device=device)

    idx = 0
    for code, size in zip(task_codes, batch_sizes):
        if size <= 0:
            continue
        S = set(itertools.chain.from_iterable(subsets[c] for c in code))
        x[idx:idx+size, code] = 1
        slice_bits = bits[idx:idx+size][:, sorted(S)]
        y[idx:idx+size] = slice_bits.sum(dim=1).remainder(2).to(torch.int64)
        idx += size

    return x, y
