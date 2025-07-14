import numpy as np
from typing import Callable, Dict

def _identity(a):      return a
def _square(a):        return np.square(a)
def _sqrt(a):          return np.sqrt(np.abs(a))
def _exp(a):           return np.exp(a) / np.e
def _sigmoid(a):       return 1 / (1 + np.exp(-5*(a-0.5)))
def _clip(a):          return np.clip(a, 0.1, 0.9)

TRANSFORMS: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "linear":  _identity,
    "square":  _square,
    "sqrt":    _sqrt,
    "exp":     _exp,
    "sigmoid": _sigmoid,
    "clip":    _clip,
}

def transform_action(action: np.ndarray, method: str) -> np.ndarray:
    """Return transformed (0-1) action; fall back to identity."""
    return TRANSFORMS.get(method, _identity)(action)
