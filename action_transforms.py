import numpy as np

def transform_action(action, method='linear', timestep=0.0):
    if method == 'linear':
        return action
    elif method == 'square':
        return np.square(action)
    elif method == 'sqrt':
        return np.sqrt(np.abs(action))
    elif method == 'exp':
        return np.exp(action) / np.exp(1)
    elif method == 'sigmoid':
        return 1 / (1 + np.exp(-5 * (action - 0.5)))
    elif method == 'clip':
        return np.clip(action, 0.1, 0.9)
    else:
        return action
