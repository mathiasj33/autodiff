import contextlib

compute_grads = True

@contextlib.contextmanager
def no_grad():
    global compute_grads
    compute_grads = False
    yield
    compute_grads = True
