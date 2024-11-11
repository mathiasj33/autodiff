from torch_geometric.datasets import Planetoid
import torch
from torch.nn import CrossEntropyLoss
from tqdm import trange
from losses import batch_cross_entropy
from autodiff.functional import *

torch.manual_seed(42)


class GCNLayer:
    def __init__(self, num_in, num_out, degs):
        self.num_in = num_in
        self.num_out = num_out
        self.degs = degs
        self.W = torch.zeros(num_out, num_in, requires_grad=True)
        torch.nn.init.kaiming_normal_(self.W, nonlinearity='relu')
        self.b = torch.zeros(num_out, requires_grad=True)

    def __call__(self, features, edge_index):
        # features: nodes x num_in
        scaled_features = features / torch.sqrt(self.degs * 2).reshape(-1, 1)
        msgs = features[edge_index[0]] / torch.sqrt(
            self.degs[edge_index[1]] + self.degs[edge_index[0]]).reshape(-1, 1)
        result = scaled_features.scatter_add(0, edge_index[1].reshape(-1, 1).expand(msgs.shape), msgs)
        return result @ self.W.T + self.b

    def parameters(self):
        return [self.W, self.b]


class GCN:
    def __init__(self, num_in, num_out, degs):
        self.conv1 = GCNLayer(num_in, 16, degs)
        self.conv2 = GCNLayer(16, num_out, degs)

    def __call__(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = relu(x)
        x = self.conv2(x, edge_index)

        return x

    def parameters(self):
        return self.conv1.parameters() + self.conv2.parameters()


def main():
    dataset = Planetoid(root='data/Cora', name='Cora')
    data = dataset[0]
    degs = torch.zeros((data.num_nodes,), dtype=torch.int64)
    degs.scatter_add_(0, data.edge_index[0], torch.ones_like(data.edge_index[0]))

    model = GCN(dataset.num_features, dataset.num_classes, degs)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    num_epochs = 10

    for _ in trange(num_epochs):
        optimizer.zero_grad()
        out = model(data)
        loss = batch_cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

    pred = model(data).argmax(dim=1)
    print(f'Train accuracy: {(pred[data.train_mask] == data.y[data.train_mask]).float().mean():.3f}')
    print(f'Test accuracy: {(pred[data.test_mask] == data.y[data.test_mask]).float().mean():.3f}')


if __name__ == '__main__':
    main()
