import argparse
import importlib
import logging
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
from tqdm import trange

from syntheticChrissAlmgren_extended import MarketEnvironment
from ddpg_agent import Agent


def make_env(env_name: str, seed: int) -> MarketEnvironment:
    if env_name == "ac_default":
        return MarketEnvironment(randomSeed=seed)
    # Dynamically import alternative envs if needed, e.g. gbm_env.GBMMarketEnvironment
    try:
        module_name, cls_name = env_name.split(":")
        mod = importlib.import_module(module_name)
        cls = getattr(mod, cls_name)
        return cls(randomSeed=seed)
    except ValueError:
        raise ValueError(f"Unknown env format '{env_name}'. Use 'ac_default' or 'module:ClassName'.")


def make_agent(agent_name: str, state_dim: int, action_dim: int, seed: int):
    if agent_name == "ddpg":
        return Agent(state_size=state_dim, action_size=action_dim, random_seed=seed)
    try:
        module_name, cls_name = agent_name.split(":")
        mod = importlib.import_module(module_name)
        cls = getattr(mod, cls_name)
        return cls(state_size=state_dim, action_size=action_dim, random_seed=seed)
    except ValueError:
        raise ValueError(
            f"Unknown agent '{agent_name}'. Use 'ddpg' or 'module:ClassName'.")


def train(env_name: str, agent_name: str, reward_fn: str, episodes: int, seed: int,
          noiseflag: bool = True):
    log = logging.getLogger(reward_fn)
    env = make_env(env_name, seed)
    agent = make_agent(agent_name, env.observation_space_dimension(), env.action_space_dimension(), seed)

    ep_rewards = []
    ep_shortfalls = []
    for ep in trange(episodes, desc=f"{agent_name}:{reward_fn}"):
        state = env.reset(seed + ep)
        env.start_transactions()
        agent.reset()
        total_reward = 0.0
        done = False
        while not done:
            action = agent.act(state, add_noise=noiseflag)
            next_state, reward, done, info = env.step(action, reward_function=reward_fn)
            reward_scalar = reward.item()
            agent.step(state, action, reward_scalar, next_state, done)
            total_reward += reward_scalar
            state = next_state
        ep_rewards.append(total_reward)
        ep_shortfalls.append(info.implementation_shortfall)
        if (ep + 1) % 1000 == 0:
            log.info(f"Ep {ep + 1}/{episodes}: totReward={total_reward:.4f}  shortfall=${info.implementation_shortfall:,.0f}")

    df = pd.DataFrame({
        "episode": list(range(episodes)),
        "reward": ep_rewards,
        "shortfall": ep_shortfalls
    })
    out_file = Path(f"runs/{agent_name}_{reward_fn}.csv")
    out_file.parent.mkdir(exist_ok=True)
    df.to_csv(out_file, index=False)

    return np.mean(ep_rewards), np.std(ep_rewards), np.mean(ep_shortfalls), np.std(ep_shortfalls)


def main(argv: List[str] | None = None):
    parser = argparse.ArgumentParser(description="Run RL agent on trading env with flexible rewards.")
    parser.add_argument("--agent", default="ddpg")
    parser.add_argument("--env", default="ac_default")
    parser.add_argument("--reward", nargs="+", default=["ac_utility"])
    parser.add_argument("--episodes", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no_noise", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    summary = {}
    for r in args.reward:
        mean_r, std_r, mean_s, std_s = train(
            env_name=args.env,
            agent_name=args.agent,
            reward_fn=r,
            episodes=args.episodes,
            seed=args.seed,
            noiseflag=not args.no_noise,
        )
        summary[r] = {
            "reward_mean": mean_r,
            "reward_std": std_r,
            "shortfall_mean": mean_s,
            "shortfall_std": std_s,
        }
        logging.info(f"[DONE] {r}: shortfall_mean=${mean_s:,.0f}")

    pd.DataFrame(summary).T.to_csv("runs/summary.csv")


if __name__ == "__main__":
    main()
