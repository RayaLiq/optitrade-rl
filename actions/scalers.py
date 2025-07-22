from __future__ import annotations

import numpy as np
from typing import Any, Callable, Dict

def _identity(a, env):
    return np.clip(a, 0.0, 1.0)

def _square(a, env):
    return np.clip(np.square(a), 0.0, 1.0)

def _sqrt(a, env):
    return np.clip(np.sqrt(np.abs(a)), 0.0, 1.0)

def _exp(a, env):
    return np.clip(np.exp(a) / np.e, 0.0, 1.0)  # ~[0,1]

def _sigmoid(a, env):
    return 1 / (1 + np.exp(-5 * (a - 0.5)))

def _clip(a, env):
    return np.clip(a, 0.1, 0.9)


def trading_rate(a, env, max_rate: float = 0.10):
    """Map raw a → fraction of ADV to trade now, converted to 0‑1 of inventory."""
    adv_shares = max_rate * env.dtv * a  # absolute shares this day
    return np.clip(adv_shares / env.shares_remaining, 0, 1)

def kappa_schedule(a, env):
    """Front‑load via AC κ schedule: f(t)=sinh[κ(T−t)] / sinh(κT).*a"""
    baseline = np.sinh(env.kappa * env.timeHorizon * env.tau) / np.sinh(env.kappa * env.liquidation_time)
    return np.clip(a * baseline, 0, 1)


try:
    import torch
    import torch.nn as nn

    class _ActionShaper(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 1), nn.Sigmoid()
            )
        def forward(self, a, t):
            x = torch.cat([a, t], dim=-1)
            return self.net(x)

    _learned_shaper = _ActionShaper()

    def learned_transform(a, env):
        if not isinstance(a, np.ndarray):
            a = np.array([a])
        a_t = torch.tensor(a, dtype=torch.float32).view(1, 1)
        t_t = torch.tensor([[env.k / env.num_n]], dtype=torch.float32)
        with torch.no_grad():
            out = _learned_shaper(a_t, t_t).item()
        return float(np.clip(out, 0, 1))
except ImportError:  # no torch
    def learned_transform(a, env):  # type: ignore
        return np.clip(a, 0, 1)


def adaptive_transform(a, env):
    t = env.k / env.num_n
    if t < 0.33:      # early
        return np.square(a)
    elif t < 0.66:    # mid
        return 0.5 * (a + np.square(a))
    else:             # late
        return np.clip(a, 0.05, 0.25)

def momentum_aware(a, env):
    momentum = np.mean(list(env.logReturns)) if env.logReturns else 0.0
    if momentum < -1e-3:        # down‑trend
        return min(1.0, 2 * np.square(a))
    elif momentum > 1e-3:       # up‑trend
        return np.sqrt(a)
    return a

def adaptive_smart(a, env):
    t = env.k / env.num_n
    adaptive = np.square(a) if t < 0.5 else np.clip(a, 0.05, 0.25)
    smooth   = np.square(a) * (1.0 - t + 0.1)
    return np.clip(0.6 * adaptive + 0.4 * smooth, 0, 1)

def poly_decay(a, env, p: float = 1.7):
    t = env.k / env.num_n
    scaled = (1 - t) * (a ** p) + t * a
    return np.clip(scaled, 0, 1)

def inverse_kappa(a, env):
    k_norm = np.clip((env.kappa - 0.001) / 0.009, 0, 1)
    blend = k_norm * np.square(a) + (1 - k_norm) * a
    return np.clip(blend, 0, 1)



TRANSFORMS: Dict[str, Callable[[np.ndarray, Any], np.ndarray]] = {
    "linear":   _identity,
    "square":   _square,
    "sqrt":     _sqrt,
    "exp":      _exp,
    "sigmoid":  _sigmoid,
    "clip":     _clip,
    "trading_rate":   trading_rate,
    "kappa_sched":    kappa_schedule,
    "adaptive":       adaptive_transform,
    "momentum":       momentum_aware,
    "adaptive_smart": adaptive_smart,
    "poly_decay":     poly_decay,
    "inverse_kappa":  inverse_kappa,
    "learned":        learned_transform,
}

def transform_action(action: np.ndarray | float, env, method: str = "linear") -> float:
    if not isinstance(action, np.ndarray):
        action = np.array(action, dtype=np.float32)
    fn = TRANSFORMS.get(method, _identity)
    return float(fn(action, env))
