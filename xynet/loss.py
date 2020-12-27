import numpy as np

from xynet.tensor import Tensor


class Loss:
    def loss(self):
        raise NotImplementedError

    def gradient(self):
        raise NotImplementedError


class SSE(Loss):
    def loss(self, pred: Tensor, actual: Tensor) -> float:
        return np.sum((pred - actual) ** 2)

    def gradient(self, pred: Tensor, actual: Tensor) -> float:
        return 2 * (pred - actual)
