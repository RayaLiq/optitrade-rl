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


def inv_time_penalty(env, info, action=None):
    """Encourages finishing earlier. Penalty increases over time."""
    inv_frac = env.shares_remaining / env.total_shares
    time_frac = 1.0 - env.timeHorizon / env.num_n
    penalty = time_frac * inv_frac
    return -penalty

def risk_adjusted_utility(env, info, action=None):
    """AC-style utility with explicit risk adjustment for remaining shares."""
    delta_u = (abs(env.prevUtility) - abs(env.compute_AC_utility(env.shares_remaining))) / abs(env.prevUtility)
    risk_term = 0.5 * env.llambda * env.singleStepVariance * (env.shares_remaining / env.total_shares)
    return delta_u - risk_term


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
        if abs(env.logReturns[0]) > 1e-6:
            mom = np.log(env.prevPrice / np.exp(env.logReturns[0]))
        else:
            mom = 0.0
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


def Implementation_Shortfall_Focused(env, info, action=None):
    # Calculate incremental implementation shortfall improvement
    current_value = env.shares_remaining * env.startingPrice
    new_value = env.shares_remaining * info.price
    return (current_value - new_value) / env.startingPrice 


def Implemetation_Shortfall_Time_Sensitive(env, info, action=None):
    current_value = env.shares_remaining * env.startingPrice
    new_value = env.shares_remaining * info.price
    time_weight = 1 + 0.5*(1 - env.timeHorizon/env.num_n)  # 1x early, 1.5x late
    return time_weight * (current_value - new_value) / env.startingPrice


def permanent_impact_penalty(env, info, action=None):
    # Penalize actions that cause long-term price depression
    impact_ratio = info.currentPermanentImpact / (env.gamma * env.dtv * 0.01)
    return -impact_ratio * info.share_to_sell_now  

def VWAP_Benchmarking(env, info, action=None):
    env.cumulative_volume += info.share_to_sell_now
    env.vwap_numerator += info.share_to_sell_now * info.exec_price
    vwap = env.vwap_numerator / env.cumulative_volume
    return (info.exec_price - vwap) * info.share_to_sell_now  

def Market_Aware_Reward(env, info, action=None):
    # Incorporate market trend direction
    market_return = np.mean(list(env.logReturns))
    trend_alignment = 1 if market_return < 0 else -1  # Sell faster in downturns
    return trend_alignment * info.share_to_sell_now / env.total_shares


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
    "inv_time_penalty": inv_time_penalty,
    "risk_adjusted_utility": risk_adjusted_utility,
    "advantage_utility": advantage_utility,
    "momentum_capture": momentum_capture,
    "risk_buffered_deltaU": risk_buffered_deltaU,
    "Implementation_Shortfall_Focused": Implementation_Shortfall_Focused,
    "Implemetation_Shortfall_Time_Sensitive": Implemetation_Shortfall_Time_Sensitive,
    "permanent_impact_penalty": permanent_impact_penalty,
    "Market-Aware_Reward": Market_Aware_Reward,
    "VWAP_Benchmarking": VWAP_Benchmarking
}
