import autodiff
import numpy as np
from abc import ABC, abstractmethod


class Optimizer(ABC):
    def __init__(self, params, lr):
        self.params = params
        self.lr = lr

    def reset_grads(self):
        for param in self.params:
            param.reset_grad()

    @abstractmethod
    def step(self):
        pass


class SGD(Optimizer):
    def __init__(self, params, lr):
        super().__init__(params, lr)

    def step(self):
        with autodiff.no_grad():
            for p in self.params:
                p -= self.lr * p.grad


class Momentum(Optimizer):
    def __init__(self, params, lr, gamma=0.9):
        super().__init__(params, lr)
        self.gamma = gamma
        self.vs = [np.zeros_like(p.data) for p in params]

    def step(self):
        with autodiff.no_grad():
            for (i, p) in enumerate(self.params):
                v = self.vs[i]
                v *= self.gamma
                v += self.lr * p.grad
                p -= v


class Adam(Optimizer):
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-08):
        super().__init__(params, lr)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.ms = [np.zeros_like(p.data) for p in params]
        self.vs = [np.zeros_like(p.data) for p in params]
        self.t = 0

    def step(self):
        self.t += 1
        with autodiff.no_grad():
            for (i, p) in enumerate(self.params):
                step_size = self.lr

                m = self.ms[i]
                m *= self.beta1
                m += (1 - self.beta1) * p.grad
                step_size /= (1 - self.beta1 ** self.t)

                v = self.vs[i]
                v *= self.beta2
                v += (1 - self.beta2) * p.grad ** 2
                v_hat = np.sqrt(v) / np.sqrt(1 - self.beta2 ** self.t)

                p -= step_size * (m / (v_hat + self.eps))
