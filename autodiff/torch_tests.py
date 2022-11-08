import torch
from torch.nn.functional import cross_entropy as torch_cross_entropy
from .tensor import Tensor
import unittest
import numpy as np
import losses
from .functional import *


class TestTensor(unittest.TestCase):
    def check_expression_gradients(self, expr_fun, *vars):
        f_torch = expr_fun(*vars)
        # print(f_torch)
        f_torch.backward()
        grads_torch = [v.grad for v in vars if isinstance(v, torch.Tensor)]
        if any([torch.isnan(g.sum()) for g in grads_torch if g is not None]):
            raise ValueError('Encountered nan in torch gradient')

        vars = [Tensor(v.detach().numpy()) if isinstance(v, torch.Tensor) else v for v in vars]
        f = expr_fun(*vars)
        # print(f)
        f.backward()
        grads = [v._grad for v in vars if isinstance(v, Tensor)]

        for (g, gt) in zip(grads, grads_torch):
            if gt is None:  # requires_grad == False
                continue
            cond = np.max(np.abs(gt.detach().numpy() - g)) < 1e-2
            self.assertTrue(cond)

    def test_simple(self):
        A = torch.tensor([[1., 2],
                          [3, 4],
                          [5, 6]], requires_grad=True)
        x = torch.tensor([[1.], [2]], requires_grad=True)
        w = torch.tensor([[10.], [11], [-2]], requires_grad=True)
        f = lambda A, x, w: w.T @ (A @ x)
        self.check_expression_gradients(f, A, x, w)

    def test_complicated(self):
        #  m = 3, n = 2
        A = torch.tensor([[1., 2],
                          [3, 4],
                          [5, 6]], requires_grad=True)
        x = torch.tensor([[1.], [2]], requires_grad=True)
        w = torch.tensor([[10.], [11], [-2]], requires_grad=True)
        b = torch.tensor([[-2.], [-3.]], requires_grad=True)
        ones = torch.ones(2).reshape(-1, 1)
        f = lambda A, x, w, b, ones: w.T @ (A @ x) + b.T @ x + 4.3 * x.T @ x + (b * x).T @ ones
        self.check_expression_gradients(f, A, x, w, b, ones)

    def test_matrix(self):
        A = torch.tensor([[1., 2],
                          [3, 4],
                          [5, 6]], requires_grad=True)
        B = torch.rand((2, 5), requires_grad=True)
        f = lambda A, B: (A @ B).sum()
        self.check_expression_gradients(f, A, B)

    def test_neural(self):
        W = torch.rand((300, 40), requires_grad=True)
        x = torch.rand((40, 1), requires_grad=True)
        b = torch.rand((300, 1), requires_grad=True)
        w = torch.rand((300, 1), requires_grad=True)
        f = lambda W, x, b, w: w.T @ relu(W @ x + b)
        self.check_expression_gradients(f, W, x, b, w)

    def test_mse(self):
        x = torch.rand((5, 1), requires_grad=True)
        w = torch.randn((5, 1), requires_grad=True)
        b = torch.randn((1, 1), requires_grad=True)
        y = torch.randn((1, 1), requires_grad=True)
        f = lambda x, w, b, y: losses.mse(w.T @ x + b, y)
        self.check_expression_gradients(f, x, w, b, y)

    def test_batch_mse(self):
        X = torch.rand((7, 5), requires_grad=True)  # batch = 7
        w = torch.randn((5, 1), requires_grad=True)
        b = torch.randn((1, 1), requires_grad=True)
        y = torch.randn((7, 1), requires_grad=True)
        f = lambda X, w, b, y: losses.mse(X @ w + b, y)
        self.check_expression_gradients(f, X, w, b, y)

    def test_mse_big(self):
        x = torch.rand((5, 1), requires_grad=True)
        y = torch.randn((1, 1), requires_grad=True)
        W = torch.randn((30, 5), requires_grad=True)
        W2 = torch.randn((1, 30), requires_grad=True)
        b = torch.randn((30, 1), requires_grad=True)
        b2 = torch.randn((1, 1), requires_grad=True)
        f = lambda x, b, y, W, W2, b2: losses.mse(W2 @ (W @ x + b).relu() + b2, y)
        self.check_expression_gradients(f, x, b, y, W, W2, b2)

    def test_sum(self):
        W = torch.rand((10, 50), requires_grad=True)
        x = torch.rand((10, 1))
        f = lambda W, x: W.sum(axis=1).reshape(-1, 1).T @ x
        self.check_expression_gradients(f, W, x)

    def test_cross_entropy(self):
        x = torch.rand((50, 1), requires_grad=True)
        W = torch.rand((10, 50), requires_grad=True)
        b = torch.rand((1, 1), requires_grad=True)
        y = 4
        # print(torch_cross_entropy((W @ x + b).flatten(), torch.tensor(y)))
        f = lambda x, W, b, y: losses.cross_entropy(W @ x + b, y)
        self.check_expression_gradients(f, x, W, b, y)

    def test_batch_cross_entropy(self):
        X = torch.rand((7, 50), requires_grad=True)  # batch size of 7
        W = torch.rand((10, 50), requires_grad=True)
        b = torch.rand((1, 10), requires_grad=True)
        y = torch.tensor([3, 5, 7, 3, 5, 1, 2])
        # print(torch_cross_entropy(X @ W.T + b, y))
        f = lambda X, W, b, y: losses.batch_cross_entropy(X @ W.T + b, y)
        self.check_expression_gradients(f, X, W, b, y)

    def test_neural_big(self):
        batch = 7
        hidden = 30

        x = torch.randn((batch, 784), requires_grad=True)

        w1 = torch.randn((784, hidden), requires_grad=True)
        b1 = torch.randn((1, hidden), requires_grad=True)

        w2 = torch.randn((hidden, 10), requires_grad=True)
        b2 = torch.randn((1, 10), requires_grad=True)

        y = torch.tensor(list(range(7)))

        # print(torch_cross_entropy((x @ w1 + b1).sigmoid() @ w2 + b2, y))

        def network(x, y, w1, b1, w2, b2):
            h1 = sigmoid(x @ w1 + b1)
            o = h1 @ w2 + b2
            return losses.batch_cross_entropy(o, y)

        self.check_expression_gradients(network, x, y, w1, b1, w2, b2)

    def test_final(self):
        x = torch.rand((64, 784))
        y = torch.randint(0, 10, (64,))

        w1 = torch.normal(0, np.sqrt(2 / 784), size=(784, 100), requires_grad=True)
        b1 = torch.zeros(1, 100, requires_grad=True)

        w2 = torch.normal(0, np.sqrt(2 / 100), size=(100, 30), requires_grad=True)
        b2 = torch.zeros(1, 30, requires_grad=True)

        w3 = torch.normal(0, np.sqrt(2 / 30), size=(30, 10), requires_grad=True)
        b3 = torch.zeros(1, 10, requires_grad=True)

        def network(x, y, w1, b1, w2, b2, w3, b3):
            reg = 0.001
            h1 = relu(x @ w1 + b1)
            h2 = relu(h1 @ w2 + b2)
            o = h2 @ w3 + b3
            return losses.batch_cross_entropy(o, y) + reg * (w1.square().sum() + w2.square().sum() + w3.square().sum())

        self.check_expression_gradients(network, x, y, w1, b1, w2, b2, w3, b3)
