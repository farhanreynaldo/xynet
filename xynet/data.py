from typing import Iterator, Tuple

import numpy as np

from xynet.tensor import Tensor


class DataIterator:
    def __call__(
        self, input: Tensor, actual: Tensor
    ) -> Iterator[Tuple[Tensor, Tensor]]:
        raise NotImplementedError


class BatchIterator(DataIterator):
    def __init__(self, batch_size: int = 40) -> None:
        self.batch_size = batch_size

    def __call__(
        self, input: Tensor, actual: Tensor
    ) -> Iterator[Tuple[Tensor, Tensor]]:
        starts = np.arange(0, len(input), self.batch_size)
        np.random.shuffle(starts)
        for start in starts:
            end = start + self.batch_size
            batch_input = input[start:end]
            batch_actual = actual[start:end]
            yield batch_input, batch_actual
