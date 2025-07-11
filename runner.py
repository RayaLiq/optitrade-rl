import argparse
import importlib
import logging
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
from tqdm import trange
import seaborn as sns
import matplotlib.pyplot as plt

from rewards import REWARD_FN_MAP
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


def train_once(env_name: str, agent_name: str, reward_fn: str, episodes: int, seed: int,
               csv_dir: Path, noiseflag: bool = True):
    log = logging.getLogger(f"{agent_name}:{reward_fn}")

    env = make_env(env_name, seed)
    agent = make_agent(agent_name, env.observation_space_dimension(), env.action_space_dimension(), seed)

    rewards, shortfalls = [], []

    for ep in trange(episodes, desc=f"{agent_name}:{reward_fn}"):
        state = env.reset(seed + ep)
        env.start_transactions()
        agent.reset()
        tot_r = 0.0
        done = False
        while not done:
            action = agent.act(state, add_noise=noiseflag)
            next_state, reward, done, info = env.step(action, reward_function=reward_fn)
            reward = np.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)
            r_scalar = reward.item()
            agent.step(state, action, r_scalar, next_state, done)
            tot_r += r_scalar
            state = next_state
        rewards.append(tot_r)
        shortfalls.append(info.implementation_shortfall)
        if (ep + 1) % 1000 == 0:
            log.info("Ep %d/%d  R=%.3f  IS=$%s", ep + 1, episodes, tot_r,
                     f"{info.implementation_shortfall:,.0f}")

    # Save per‑episode CSV
    csv_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"episode": range(episodes), "reward": rewards, "shortfall": shortfalls}) \
        .to_csv(csv_dir / f"{agent_name}_{reward_fn}.csv", index=False)

    return (np.mean(rewards), np.std(rewards), np.mean(shortfalls), np.std(shortfalls))


def main(argv: List[str] | None = None):
    p = argparse.ArgumentParser("runner")
    p.add_argument("--agent", default="ddpg", help="agent name or module:Class")
    p.add_argument("--env",   default="ac_default", help="env name or module:Class")
    p.add_argument("--reward", nargs="+", default=["ac_utility"], help="rewards for single‑run mode")
    p.add_argument("--compare", nargs="*", help="Run a batch comparison of rewards")
    p.add_argument("--episodes", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--csv-dir", default="runs", help="folder to dump CSVs")
    p.add_argument("--no-noise", action="store_true", help="turn off exploration noise")
    p.add_argument("--plot", action="store_true", help="plot summary bar + box plots if comparing")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if args.compare is not None:
        if len(args.compare) == 0 or args.compare[0].lower() == "all":
            batch = list(REWARD_FN_MAP.keys())
        else:
            batch = args.compare
    else:
        batch = args.reward

    csv_dir = Path(args.csv_dir)

    summary = {}
    for r in batch:
        mean_r, std_r, mean_s, std_s = train_once(
            env_name=args.env,
            agent_name=args.agent,
            reward_fn=r,
            episodes=args.episodes,
            seed=args.seed,
            csv_dir=csv_dir,
            noiseflag=not args.no_noise,
        )
        summary[r] = {
            "reward_mean": mean_r,
            "reward_std":  std_r,
            "shortfall_mean": mean_s,
            "shortfall_std":  std_s,
        }
        logging.info("[DONE] %-20s  IS_mean=$%s", r, f"{mean_s:,.0f}")

    # Save summary CSV
    pd.DataFrame(summary).T.to_csv(csv_dir / "summary_statistics.csv")

    # Optional plotting
    if args.plot and len(batch) > 1:
        df_sum = pd.DataFrame(summary).T.reset_index().rename(columns={"index": "Reward"})
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df_sum, x="Reward", y="shortfall_mean", hue="Reward", palette="coolwarm", legend=False)
        plt.ylabel("Avg Implementation Shortfall ($)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(csv_dir / "shortfall_bar.png")
        plt.close()

if __name__ == "__main__":
    main()
