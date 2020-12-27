from typing import List, Iterator, Tuple, Dict

from xynet.tensor import Tensor
import numpy as np


class Layer:
    def __init__(self):
        self.parameters: Dict[str, Tensor] = {}
        self.gradients: Dict[str, Tensor] = {}

    def forward(self, input: Tensor) -> Tensor:
        raise NotImplementedError

    def backward(self, input: Tensor) -> Tensor:
        raise NotImplementedError


class ReLU(Layer):
    def forward(self, input: Tensor) -> Tensor:
        self.input = input
        return input * (input > 0)

    def backward(self, gradient: Tensor) -> Tensor:
        return 1.0 * (gradient > 0)


class Sigmoid(Layer):
    def forward(self, input: Tensor) -> Tensor:
        self.input = input
        return 1 / (1 + np.exp(-input))

    def backward(self, gradient: Tensor) -> Tensor:
        sigmoid = 1 / (1 + np.exp(-gradient))
        return sigmoid * (1 - sigmoid)


class Linear(Layer):
    def __init__(self, n_input: int, n_output: int) -> None:
        super().__init__()
        self.parameters["weights"] = np.random.randn(n_input, n_output) - 0.5
        self.parameters["bias"] = np.random.randn(n_output) - 0.5

    def forward(self, input: Tensor) -> Tensor:
        """
        Perform matrix multiplication such as X @ W + b,
        where X is our input matrix, W is weights, and b
        is bias.
        """
        self.input = input
        return input @ self.parameters["weights"] + self.parameters["bias"]

    def backward(self, gradient: Tensor) -> Tensor:
        self.gradients["bias"] = np.sum(gradient, axis=0)
        self.gradients["weights"] = self.input.T @ gradient
        return gradient @ self.parameters["weights"].T


class Sequential(Layer):
    def __init__(self, layers: List[Layer]) -> None:
        self.layers = layers

    def add(self, layer: Layer) -> None:
        self.layers.append(layer)

    def forward(self, input: Tensor) -> Tensor:
        for layer in self.layers:
            input = layer.forward(input)
        return input

    def backward(self, gradient: Tensor) -> Tensor:
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)
        return gradient

    def parameters_and_gradients(self) -> Iterator[Tuple[Tensor, Tensor]]:
        for layer in self.layers:
            for name, parameter in layer.parameters.items():
                gradient = layer.gradients[name]
                yield parameter, gradient
