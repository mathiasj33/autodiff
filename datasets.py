from torchvision import datasets, transforms
from globals import memory
from sklearn.utils import shuffle


class Dataset:
    def __init__(self, x_train, y_train, x_val, y_val, x_test, y_test, batch_size=64):
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.x_test = x_test
        self.y_test = y_test
        self.batch_size = batch_size
        self.num_batches = x_train.shape[0] // self.batch_size + 1

    def training_batches(self):
        x, y = shuffle(self.x_train, self.y_train)
        for i in range(self.num_batches):
            start = i * self.batch_size
            end = min(start + self.batch_size, x.shape[0])
            yield x[start: end], y[start: end]


@memory.cache
def load_mnist():
    train = datasets.MNIST('../data', train=True, download=True, transform=transforms.ToTensor())
    test = datasets.MNIST('../data', train=False, download=True, transform=transforms.ToTensor())
    x_train = train.data.reshape(-1, 28 * 28).float().numpy()
    x_train /= 255
    y_train = train.targets.numpy()

    x_test = test.data.reshape(-1, 28 * 28).float().numpy()
    x_test /= 255
    y_test = test.targets.numpy()

    return Dataset(x_train[:50000], y_train[:50000], x_train[50000:], y_train[50000:], x_test, y_test, batch_size=128)
