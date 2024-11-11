from torch_geometric.datasets import Planetoid
from tqdm import trange
from losses import batch_cross_entropy
import numpy as np
from autodiff.tensor import Tensor
from autodiff.functional import *
from optimizers import *

np.random.seed(42)


class GCNLayer:
    def __init__(self, num_in, num_out, degs):
        self.num_in = num_in
        self.num_out = num_out
        self.degs = degs
        self.W = self.init_he(num_in, num_out).T
        self.b = Tensor(np.zeros((1, num_out), dtype=np.float32))

    @staticmethod
    def init_he(n_in, n_out):
        return Tensor(np.random.normal(0, np.sqrt(2 / n_in), (n_in, n_out)).astype(np.float32))

    def __call__(self, features, edge_index):
        # features: nodes x num_in
        scaled_features = features / sqrt(self.degs * 2).reshape(-1, 1)
        msgs = features[edge_index[0]]
        msgs = msgs / sqrt(self.degs[edge_index[1]] + self.degs[edge_index[0]]).reshape(-1, 1)
        result = scaled_features.scatter_add(edge_index[1], msgs)
        return result @ self.W.T + self.b

    def parameters(self):
        return [self.W, self.b]


class GCN:
    def __init__(self, num_in, num_out, degs):
        self.conv1 = GCNLayer(num_in, 16, degs)
        self.conv2 = GCNLayer(16, num_out, degs)

    def __call__(self, data):
        x, edge_index = Tensor(data.x, track_grad=False), Tensor(data.edge_index, track_grad=False)

        x = self.conv1(x, edge_index)
        x = relu(x)
        x = self.conv2(x, edge_index)

        return x

    def parameters(self):
        return self.conv1.parameters() + self.conv2.parameters()


def main():
    dataset = Planetoid(root='data/Cora', name='Cora')
    data = dataset[0]
    degs = np.zeros((data.num_nodes, 1), dtype=np.int32)
    np.add.at(degs, data.edge_index[0].numpy(), np.ones_like(data.edge_index[0].numpy()).reshape(-1, 1))
    degs = Tensor(degs)

    model = GCN(dataset.num_features, dataset.num_classes, degs)

    optimizer = Adam(model.parameters(), lr=0.01)
    num_epochs = 10

    for _ in trange(num_epochs):
        optimizer.reset_grads()
        out = model(data)
        loss = batch_cross_entropy(out[data.train_mask.numpy()], data.y[data.train_mask].numpy())
        loss.backward()
        optimizer.step()

    pred = model(data).to_numpy().argmax(axis=1)
    print(f'Train accuracy: {(pred[data.train_mask] == data.y[data.train_mask].numpy()).mean():.3f}')
    print(f'Test accuracy: {(pred[data.test_mask] == data.y[data.test_mask].numpy()).mean():.3f}')


if __name__ == '__main__':
    main()
