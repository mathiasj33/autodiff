from autodiff.torch_tests import TestTensor
import torch
from autodiff.functional import *
from torch_geometric.datasets import Planetoid
from losses import batch_cross_entropy
from gnn.torch_gcn import GCN as TorchGCN
import torch
import numpy as np
from autodiff.tensor import Tensor
from gnn.gcn import GCN


class GCNTests(TestTensor):
    def test_gcn_layer(self):
        features = torch.tensor([[1, -1], [4, 7], [5, 3]], dtype=torch.float)
        edge_index = torch.tensor([[0, 1, 1, 2],
                                   [1, 0, 2, 1]], dtype=torch.long)
        W = torch.zeros(2, 2, requires_grad=True)
        torch.nn.init.kaiming_normal_(W, nonlinearity='relu')
        b = torch.zeros(2, requires_grad=True)
        y = torch.tensor([0, 1, 0])
        degs = torch.tensor([1, 2, 1])

        def gcn_layer(features, edge_index, degs, W, b, y):
            scaled_features = features / sqrt(degs * 2).reshape(-1, 1)
            msgs = features[edge_index[0]] / sqrt(degs[edge_index[1]] + degs[edge_index[0]]).reshape(-1, 1)
            if isinstance(features, torch.Tensor):
                result = scaled_features.scatter_add(0, edge_index[1].reshape(-1, 1).expand(msgs.shape), msgs)
            else:
                result = scaled_features.scatter_add(edge_index[1], msgs)
            result = result @ W.T + b
            return batch_cross_entropy(result, y)

        self.check_expression_gradients(gcn_layer, features, edge_index, degs, W, b, y)

    def test_gcn(self):
        dataset = Planetoid(root='data/Cora', name='Cora')
        data = dataset[0]
        features = torch.tensor([[1, -1], [4, 7], [5, 3]], dtype=torch.float)
        edge_index = torch.tensor([[0, 1, 1, 2],
                                   [1, 0, 2, 1]], dtype=torch.long)
        data.x = features
        data.edge_index = edge_index
        data.num_nodes = 3
        data.num_features = 2
        data.num_classes = 2
        data.y = torch.tensor([0, 1, 0])
        data.train_mask = torch.tensor([True, True, False])

        degs = np.zeros((data.num_nodes, 1), dtype=np.int32)
        np.add.at(degs, data.edge_index[0].numpy(), np.ones_like(data.edge_index[0].numpy()).reshape(-1, 1))
        degs = Tensor(degs)

        model = GCN(data.num_features, data.num_classes, degs)
        torchmodel = TorchGCN(data.num_features, data.num_classes, torch.tensor(degs.data))
        model.conv1.W.data = torchmodel.conv1.W.detach().numpy()
        model.conv2.W.data = torchmodel.conv2.W.detach().numpy()

        out1 = model(data)
        loss1 = batch_cross_entropy(out1[data.train_mask.numpy()], data.y[data.train_mask].numpy())
        out2 = torchmodel(data)
        loss2 = batch_cross_entropy(out2[data.train_mask], data.y[data.train_mask])
        loss1.backward()
        loss2.backward()

        grads_torch = [v.grad for v in torchmodel.parameters() if isinstance(v, torch.Tensor)]
        grads = [v._grad for v in model.parameters() if isinstance(v, Tensor)]
        pass
