from typing import List
from xynet.layer import Layer


class Optimizer:
    def step(self):
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, learning_rate: float = 0.01) -> None:
        self.learning_rate = learning_rate

    def step(self, model: List[Layer]) -> None:
        for parameter, gradient in model.parameters_and_gradients():
            parameter -= self.learning_rate * gradient
