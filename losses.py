def mse(pred, y):
    return (pred - y).square().mean()

def cross_entropy(preds, y):
    return -preds[y] + preds.exp().sum().log()

def batch_cross_entropy(preds, y):
    return (-preds.gather(1, y.reshape(-1, 1)) + preds.exp().sum(axis=1).log()).mean()
