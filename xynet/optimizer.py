from typing import List
from xynet.layer import Layer
import numpy as np


class Optimizer:
    def step(self):
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, learning_rate: float = 0.01) -> None:
        self.learning_rate = learning_rate

    def step(self, model: List[Layer]) -> None:
        for parameter, gradient in model.parameters_and_gradients():
            parameter -= self.learning_rate * gradient


class Adam(Optimizer):
    def __init__(
        self,
        alpha: float = 0.01,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ):
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = 0
        self.v = 0
        self.t = 0

    def step(self, model: List[Layer]) -> None:
        for parameter, gradient in model.parameters_and_gradients():
            self.t = self.t + 1
            self.m = self.beta1 * self.m + (1 - self.beta1) * gradient
            self.v = self.beta2 * self.v + (1 - self.beta2) * (gradient ** 2)
            m_hat = self.m / (1 - self.beta1 ** self.t)
            v_hat = self.v / (1 - self.beta2 ** self.t)
            parameter -= self.alpha * (m_hat / (np.sqrt(v_hat) - self.epsilon))
