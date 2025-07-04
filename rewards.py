from __future__ import annotations

import numpy as np
from typing import Callable, Dict, Any

def _step_shortfall(env, info, action=None) -> float:
    """Cost ( >0 ) incurred on this step, normalised to [0,1]."""
    return (env.startingPrice - info.exec_price) * info.share_to_sell_now / (
        env.total_shares * env.startingPrice
    )

def ac_utility(env, info, action=None):
    """Dense utility delta from Almgren‑Chriss."""
    return (abs(env.prevUtility) - abs(env.compute_AC_utility(env.shares_remaining))) / abs(env.prevUtility)


def capture(env, info, action=None):
    """Immediate capture (positive when you earn more)."""
    return info.share_to_sell_now * info.exec_price / (env.total_shares * env.startingPrice)


def custom_penalty(env, info, action=None):
    penalty = 1e-5 * (info.share_to_sell_now ** 2)
    return (info.share_to_sell_now * info.exec_price - penalty) / (env.total_shares * env.startingPrice)


def final_shortfall(env, info, action):
    if info.done:
        return -info.implementation_shortfall / (env.total_shares * env.startingPrice)
    return 0.0


def stepwise_shortfall(env, info, action=None):
    return -_step_shortfall(env, info)


def hybrid_shortfall_risk(env, info, action=None):
    risk_penalty = 1e-7 * (env.shares_remaining ** 2)
    return -_step_shortfall(env, info) - risk_penalty


_last_action = 0.0

def smoothness_penalty(env, info, action=None):
    global _last_action
    smooth_penalty = 1e-2 * abs(env.last_trade - _last_action) if hasattr(env, "last_trade") else 0.0
    _last_action = env.last_trade if hasattr(env, "last_trade") else 0.0
    return -_step_shortfall(env, info) - smooth_penalty


def baseline_relative(env, info, action):
    if info.done:
        ac_shortfall = env.get_AC_expected_shortfall(env.total_shares)
        return (ac_shortfall - info.implementation_shortfall) / ac_shortfall
    return 0.0


def advantage_utility(env, info, action=None):
    """Per‑step advantage over AC baseline expected cost."""
    step_sf = _step_shortfall(env, info)
    base_now = env.get_AC_expected_shortfall(env.shares_remaining + info.share_to_sell_now)
    base_next = env.get_AC_expected_shortfall(env.shares_remaining)
    baseline_step = (base_next - base_now) / (env.total_shares * env.startingPrice)
    return (baseline_step - step_sf) / (abs(baseline_step) + 1e-8)


def momentum_capture(env, info, action=None, kappa: float = 1.5, alpha: float = 0.2):
    step_sf = _step_shortfall(env, info)
    if len(env.logReturns) >= 6:
        mom = np.log(env.prevPrice / env.logReturns[0])
    else:
        mom = 0.0
    inv_frac = env.shares_remaining / env.total_shares
    momentum_term = kappa * (-mom) * (info.share_to_sell_now / env.total_shares)
    return -step_sf + momentum_term - alpha * inv_frac ** 2


def risk_buffered_deltaU(env, info, action=None):
    delta_u = (abs(env.prevUtility) - abs(env.compute_AC_utility(env.shares_remaining))) / abs(env.prevUtility)
    inv_frac = env.shares_remaining / env.total_shares
    gamma = 0.5 * env.llambda / env.tau
    return delta_u - gamma * env.singleStepVariance * inv_frac


RewardFn = Callable[..., float]
REWARD_FN_MAP: Dict[str, RewardFn] = {
    "ac_utility": ac_utility,
    "capture": capture,
    "custom_penalty": custom_penalty,
    "final_shortfall": final_shortfall,
    "stepwise_shortfall": stepwise_shortfall,
    "hybrid_shortfall_risk": hybrid_shortfall_risk,
    "smoothness_penalty": smoothness_penalty,
    "baseline_relative": baseline_relative,
    "advantage_utility": advantage_utility,
    "momentum_capture": momentum_capture,
    "risk_buffered_deltaU": risk_buffered_deltaU,
}
