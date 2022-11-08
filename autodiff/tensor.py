import numpy as np
from . import context


class Tensor:
    """
    Stores a numpy array and its gradient.
    """

    def __init__(self, data, children=None, op='', name='', track_grad=True):
        if isinstance(data, float):
            self.data = np.array(data, dtype=np.float32)
        elif isinstance(data, np.ndarray):
            dtype = np.float32 if data.dtype == np.float64 else data.dtype
            self.data = data.astype(dtype)
        else:
            self.data = np.array(data)
        self._grad = 0
        self.track_grad = track_grad
        self.backward_op = lambda: None
        self.children = [] if children is None else children
        self.op = op
        self.name = name

    @property
    def grad(self):
        if not self.track_grad:
            raise ValueError('Tried accessing non-tracked gradient')
        return self._grad

    def reset_grad(self):
        self._grad = 0

    def is_vector(self):
        return 1 in self.data.shape

    @property
    def shape(self):
        return self.data.shape

    @staticmethod
    def as_tensor(array):
        return array if isinstance(array, Tensor) else Tensor(array, track_grad=False)

    def to_numpy(self):
        return self.data

    def item(self):
        return self.data.item()

    def handle_broadcasting(self, grad):
        if grad.shape == self.shape:
            return grad
        axes = tuple([i for i, dim in enumerate(self.shape) if dim != grad.shape[i]])
        return np.sum(grad, axis=axes, keepdims=True)

    def add_grad(self, grad):
        """
        Adds grad to self.grad, taking into account a possible batch dimension in grad.
        """
        self._grad += self.handle_broadcasting(grad)

    @property
    def T(self):
        if not context.compute_grads:
            return Tensor(self.data.T, track_grad=False)

        out = Tensor(self.data.T, [self], 'T', track_grad=self.track_grad)

        def backward():
            self.add_grad(out._grad.T)

        out.backward_op = backward
        return out

    def __add__(self, other):
        other = Tensor.as_tensor(other)
        if not context.compute_grads:
            return Tensor(self.data + other.data, track_grad=False)

        out = Tensor(self.data + other.data, [self, other], '+', track_grad=self.track_grad or other.track_grad)

        def backward():
            if self.track_grad:
                self.add_grad(out._grad)
            if other.track_grad:
                other.add_grad(out._grad)

        out.backward_op = backward
        return out

    def __radd__(self, other):
        return self + other

    def __iadd__(self, other):
        if context.compute_grads and self.track_grad:
            raise ValueError('In place operation on tensor with tracked gradient')
        self.data += Tensor.as_tensor(other).to_numpy()
        return self

    def __sub__(self, other):
        other = Tensor.as_tensor(other)
        if not context.compute_grads:
            return Tensor(self.data - other.data, track_grad=False)

        out = Tensor(self.data - other.data, [self, other], '-', track_grad=self.track_grad or other.track_grad)

        def backward():
            if self.track_grad:
                self.add_grad(out._grad)
            if other.track_grad:
                other.add_grad(-out._grad)

        out.backward_op = backward
        return out

    def __rsub__(self, other):
        return Tensor.as_tensor(other) - self

    def __isub__(self, other):
        if context.compute_grads and self.track_grad:
            raise ValueError('In place operation on tensor with tracked gradient')
        self.data -= Tensor.as_tensor(other).to_numpy()
        return self

    def __mul__(self, other):
        other = Tensor.as_tensor(other)
        if not context.compute_grads:
            return Tensor(self.data * other.data, track_grad=False)

        out = Tensor(self.data * other.data, [self, other], '*', track_grad=self.track_grad or other.track_grad)

        def backward():
            if self.track_grad:
                self.add_grad(out._grad * other.data)
            if other.track_grad:
                other.add_grad(out._grad * self.data)

        out.backward_op = backward
        return out

    def __rmul__(self, other):
        return self * other

    def __imul__(self, other):
        if context.compute_grads and self.track_grad:
            raise ValueError('In place operation on tensor with tracked gradient')
        self.data *= Tensor.as_tensor(other).to_numpy()
        return self

    def __matmul__(self, other):
        other = Tensor.as_tensor(other)
        if not context.compute_grads:
            return Tensor(self.data @ other.data, track_grad=False)

        out = Tensor(self.data @ other.data, [self, other], '@', track_grad=self.track_grad or other.track_grad)

        def backward():
            if self.track_grad:
                self.add_grad(out._grad @ other.data.T)
            if other.track_grad:
                other.add_grad(self.data.T @ out._grad)

        out.backward_op = backward
        return out

    def __rmatmul__(self, other):
        return Tensor.as_tensor(other) @ self

    def __imatmul__(self, other):
        if context.compute_grads and self.track_grad:
            raise ValueError('In place operation on tensor with tracked gradient')
        self.data @= Tensor.as_tensor(other).to_numpy()
        return self

    def __neg__(self):
        if not context.compute_grads:
            return Tensor(-self.data, track_grad=False)

        out = Tensor(-self.data, [self], 'neg', track_grad=self.track_grad)

        def backward():
            self.add_grad(-out._grad)

        out.backward_op = backward
        return out

    def square(self):
        return self * self

    def mean(self):
        if not context.compute_grads:
            return Tensor(self.data.mean(), track_grad=False)

        out = Tensor(self.data.mean(), [self], 'mean', track_grad=self.track_grad)

        def backward():
            self.add_grad(out._grad.item() * np.ones_like(self.data) / np.prod(self.shape))

        out.backward_op = backward
        return out

    def sum(self, axis=None):
        if not context.compute_grads:
            return Tensor(self.data.sum(axis=axis, keepdims=True), track_grad=False)

        out = Tensor(self.data.sum(axis=axis, keepdims=True), [self], 'sum', track_grad=self.track_grad)

        def backward():
            if isinstance(self._grad, int):
                self._grad = np.zeros_like(self.data)  # fix broadcasting
            self._grad += out._grad

        out.backward_op = backward
        return out

    def exp(self):
        if not context.compute_grads:
            return Tensor(np.exp(self.data), track_grad=False)

        out = Tensor(np.exp(self.data), [self], 'exp', track_grad=self.track_grad)

        def backward():
            self.add_grad(out._grad * out.data)

        out.backward_op = backward
        return out

    def log(self):
        if not context.compute_grads:
            return Tensor(np.log(self.data), track_grad=False)

        out = Tensor(np.log(self.data), [self], 'log', track_grad=self.track_grad)

        def backward():
            self.add_grad(out._grad / self.data)

        out.backward_op = backward
        return out

    def __getitem__(self, item):
        assert self.is_vector() and isinstance(item, int)
        if not context.compute_grads:
            return Tensor(self.data[item], track_grad=False)

        out = Tensor(self.data[item], [self], f'[{item}]', track_grad=self.track_grad)

        def backward():
            self._grad[[item]] += self.handle_broadcasting(out._grad)

        out.backward_op = backward
        return out

    def relu(self):
        mask = self.data > 0
        if not context.compute_grads:
            return Tensor(self.data * mask, track_grad=False)

        out = Tensor(self.data * mask, [self], 'relu', track_grad=self.track_grad)

        def backward():
            self.add_grad(out._grad * mask)

        out.backward_op = backward
        return out

    def sigmoid(self):
        sig = 1 / (1 + np.exp(-self.data))
        if not context.compute_grads:
            return Tensor(sig, track_grad=False)

        out = Tensor(sig, [self], 'sigmoid', track_grad=self.track_grad)

        def backward():
            self.add_grad(out._grad * sig * (1 - sig))

        out.backward_op = backward
        return out

    def reshape(self, *newshape):
        if not context.compute_grads:
            return Tensor(self.data.reshape(newshape), track_grad=False)

        out = Tensor(self.data.reshape(newshape), [self], 'reshape', track_grad=self.track_grad)

        def backward():
            self._grad += out._grad.reshape(self.shape)

        out.backward_op = backward
        return out

    def gather(self, axis, index):
        index = index.data  # don't backpropagate wrt index
        if not context.compute_grads:
            return Tensor(np.take_along_axis(self.data, index, axis), track_grad=False)

        out = Tensor(np.take_along_axis(self.data, index, axis), [self], 'gather', track_grad=self.track_grad)

        def backward():
            values = np.take_along_axis(self._grad, index, axis)
            assert out._grad.shape == values.shape
            np.put_along_axis(self._grad, index, values + out._grad, axis)

        out.backward_op = backward
        return out

    def backward(self):
        topo = []
        stack = [self]
        visited = set()
        while len(stack) != 0:
            current = stack[-1]
            not_visited = [v for v in current.children if v not in visited]
            if len(not_visited) > 0:
                stack.append(not_visited[0])
                continue
            stack.pop()
            topo.append(current)
            visited.add(current)

        self._grad = np.ones((1, 1), dtype=np.float32)
        for v in reversed(topo):
            if v.track_grad:
                v.backward_op()

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self._grad}, op={self.op})"
