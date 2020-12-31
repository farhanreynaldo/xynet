from xynet.layer import Linear, ReLU, Sequential
from xynet.loss import MSE
from xynet.optimizer import SGD, Adam
from xynet.data import BatchIterator

from typing import List
import numpy as np
import matplotlib.pyplot as plt

N = 1000  # num_samples_per_class
D = 1  # dimensions
C = 1  # num_classes
H = 100  # num_hidden_units

# dummy data
X = np.linspace(-1, 1, 100).reshape(-1, 1)
y = np.power(X, 3) + 0.3 * np.random.random(X.shape)

model = Sequential([Linear(D, H), ReLU(), Linear(H, C)])

criterion = MSE()
optimizer = SGD(learning_rate=1e-4)

for t in range(500):
    y_pred = model.forward(X)
    loss = criterion.loss(y_pred, y)
    print("[EPOCH]: %i, [LOSS or MSE]: %.6f" % (t, loss))
    grad = criterion.gradient(y_pred, y)
    model.backward(grad)
    optimizer.step(model)
