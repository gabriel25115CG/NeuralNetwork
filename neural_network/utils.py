def mean_squared_error(y_true, y_pred):
    n = len(y_true)
    return sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred)) / n

def mean_absolute_error(y_true, y_pred):
    """Calcule l'erreur absolue moyenne"""
    n = len(y_true)
    return sum(abs(yt - yp) for yt, yp in zip(y_true, y_pred)) / n

def r2_score(y_true, y_pred):
    """Calcule le coefficient de détermination R²"""
    y_mean = sum(y_true) / len(y_true)
    ss_tot = sum((yt - y_mean) ** 2 for yt in y_true)
    ss_res = sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred))
    
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    
    return 1 - (ss_res / ss_tot)

def initialize_weights(n_inputs, n_neurons):
    import random
    return [[random.uniform(-1,1) for _ in range(n_inputs)] for _ in range(n_neurons)]
