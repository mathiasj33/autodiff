import numpy as np
from autodiff.tensor import Tensor
from autodiff.functional import *
from losses import batch_cross_entropy
from tqdm import tqdm, trange
from datasets import load_mnist
import matplotlib.pyplot as plt
import matplotlib
from optimizers import *

np.random.seed(42)
matplotlib.rcParams['figure.dpi'] = 150
matplotlib.rcParams['backend'] = 'TkAgg'


class MnistNetwork:
    def __init__(self, hidden1=100, hidden2=70):
        self.w1 = MnistNetwork.init_he(784, hidden1)
        self.b1 = Tensor(np.zeros((1, hidden1), dtype=np.float32))

        self.w2 = MnistNetwork.init_he(hidden1, hidden2)
        self.b2 = Tensor(np.zeros((1, hidden2), dtype=np.float32))

        self.w3 = MnistNetwork.init_he(hidden2, 10)
        self.b3 = Tensor(np.zeros((1, 10), dtype=np.float32))

    @staticmethod
    def init_he(n_in, n_out):
        return Tensor(np.random.normal(0, np.sqrt(2 / n_in), (n_in, n_out)).astype(np.float32))

    def __call__(self, x):
        h1 = relu(x @ self.w1 + self.b1)
        h2 = relu(h1 @ self.w2 + self.b2)
        return h2 @ self.w3 + self.b3

    def parameters(self):
        return [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3]


def compute_loss_and_accuracy(model, x, y, loss_fn):
    x, y = Tensor(x, track_grad=False), Tensor(y, track_grad=False)
    preds = model(x)
    loss = loss_fn(preds, y)
    preds = preds.to_numpy().argmax(axis=1)
    acc = sum(preds == y.to_numpy()) / len(preds)
    return loss.item(), acc


def cross_entropy_l2(model, preds, y, reg):
    return batch_cross_entropy(preds, y) + reg * \
           (model.w1.square().sum() + model.w2.square().sum() + model.w3.square().sum())


def train(model, data, num_epochs):
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    iterations = []
    # optimizer = SGD(model.parameters(), lr=0.2)
    optimizer = Momentum(model.parameters(), lr=5e-2, gamma=0.9)
    loss_fn = lambda preds, y: cross_entropy_l2(model, preds, y, reg=5e-4)
    pbar = trange(num_epochs, desc='Epoch', ncols=120)

    for i in pbar:
        pbar_inner = tqdm(data.training_batches(), total=data.num_batches, desc='Batches', leave=False, ncols=120)
        for batch_idx, (x, y) in enumerate(pbar_inner):
            x, y = Tensor(x, track_grad=False), Tensor(y, track_grad=False)
            preds = model(x)
            loss = loss_fn(preds, y)

            optimizer.reset_grads()
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                with autodiff.no_grad():
                    train_loss, train_acc = compute_loss_and_accuracy(model, data.x_train, data.y_train, loss_fn)
                    train_accs.append(train_acc)
                    train_losses.append(train_loss)

                    val_loss, val_acc = compute_loss_and_accuracy(model, data.x_val, data.y_val, loss_fn)
                    val_accs.append(val_acc)
                    val_losses.append(val_loss)

                    iterations.append(i * data.num_batches + batch_idx)
                    pbar.set_postfix({'train_acc': train_acc, 'val_acc': val_acc})

    return train_losses, train_accs, val_losses, val_accs, iterations


def plot_results(train_losses, train_accs, val_losses, val_accs, iterations):
    plt.subplot(121)
    plt.gca().set_title('Accuracy')
    plt.plot(iterations, train_accs)
    plt.plot(iterations, val_accs)
    plt.legend(['Train accuracy', 'Val accuracy'])
    plt.xlabel('Training steps')
    plt.subplot(122)
    plt.gca().set_title('Loss')
    plt.plot(iterations, train_losses)
    plt.plot(iterations, val_losses)
    plt.legend(['Train loss', 'Val loss'])
    plt.xlabel('Training steps')


def main():
    data = load_mnist()
    model = MnistNetwork()

    results = train(model, data, num_epochs=5)
    test_acc = compute_loss_and_accuracy(model, data.x_test, data.y_test, lambda a, b: np.zeros(1))[1]
    print(f'Test accuracy: {test_acc:.3f}')

    plot_results(*results)
    plt.show()


if __name__ == '__main__':
    main()
