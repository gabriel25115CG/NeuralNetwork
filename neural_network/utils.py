def mean_squared_error(y_true, y_pred):
    n = len(y_true)
    return sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred)) / n

def initialize_weights(n_inputs, n_neurons):
    import random
    return [[random.uniform(-1,1) for _ in range(n_inputs)] for _ in range(n_neurons)]
