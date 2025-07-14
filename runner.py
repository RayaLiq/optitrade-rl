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
from actions import transform_action, TRANSFORMS


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


def train_once(env_name: str, agent_name: str, reward_fn: str, act_method: str, episodes: int, seed: int,
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
            raw_action = agent.act(state, add_noise=noiseflag)
            action = transform_action(raw_action, act_method)
            next_state, reward, done, info = env.step(action, reward_function=reward_fn)
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

def run_action_transform_test(transform_methods: List[str], seed_count: int, output_dir: Path):  
    results = {}
    for method in transform_methods:
        shortfalls = []
        for seed in range(seed_count):
            env = MarketEnvironment(randomSeed=seed)
            env.reset(seed=seed)
            env.start_transactions()
            agent = Agent(state_size=8, action_size=1, random_seed=seed)
            state = env.initial_state
            done = False
            while not done:
                action = agent.act(state, add_noise=False, transform_method=method)  
                next_state, reward, done, info = env.step(action)
                state = next_state
            shortfalls.append(getattr(info, "implementation_shortfall", np.nan))
        results[method] = {
            'mean_shortfall': np.nanmean(shortfalls),
            'std_shortfall': np.nanstd(shortfalls),
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results).T.reset_index().rename(columns={"index": "method"})
    df.to_csv(output_dir / "action_transform_results.csv", index=False)

    plt.figure(figsize=(10, 5))
    plt.bar(df['method'], df['mean_shortfall'], yerr=df['std_shortfall'], capsize=5, color='skyblue')
    plt.title("Comparison of Action Transformation Methods")
    plt.ylabel("Implementation Shortfall ($)")
    plt.xlabel("Transformation Method")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "action_transform_plot.png")
    plt.close()

def main(argv: List[str] | None = None):
    p = argparse.ArgumentParser("runner")
    p.add_argument("--agent", default="ddpg", help="agent name or module:Class")
    p.add_argument("--env",   default="ac_default", help="env name or module:Class")
    p.add_argument("--reward", nargs="+", default=["ac_utility"], help="rewards for single‑run mode")
    p.add_argument("--compare-reward", nargs="*", help="Batch-compare reward functions (use 'all' for every reward)")
    p.add_argument("--episodes", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--csv-dir", default="runs", help="folder to dump CSVs")
    p.add_argument("--no-noise", action="store_true", help="turn off exploration noise")
    p.add_argument("--plot", action="store_true", help="plot summary bar + box plots if comparing")
    p.add_argument("--transform-methods", nargs="+", default=["linear", "square", "sqrt", "exp", "sigmoid", "clip"], help="Action transforms")
    p.add_argument("--action", nargs="+", default=["linear"],
               help="Action transform(s) for single-run mode")
    p.add_argument("--compare-action", nargs="*",
                help="Batch compare action transforms (use 'all' for every method)")


    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    csv_dir = Path(args.csv_dir)

    # ------------------------------------------------------------------
    # Build reward and action batches
    # ------------------------------------------------------------------
    if args.compare_reward is not None:
        reward_batch = (list(REWARD_FN_MAP.keys()) if
                        (len(args.compare_reward) == 0 or
                         args.compare_reward[0].lower() == "all")
                        else args.compare_reward)
    else:
        reward_batch = args.reward

    if args.compare_action is not None:
        action_batch = (list(TRANSFORMS.keys()) if
                        (len(args.compare_action) == 0 or
                         args.compare_action[0].lower() == "all")
                        else args.compare_action)
    else:
        action_batch = args.action
    # ------------------------------------------------------------------

    summary: dict[str, dict[str, float]] = {}

    for r in reward_batch:
        for a in action_batch:
            tag = f"{r}|{a}"
            m_r, s_r, m_s, s_s = train_once(
                env_name=args.env,
                agent_name=args.agent,
                reward_fn=r,
                act_method=a,
                episodes=args.episodes,
                seed=args.seed,
                csv_dir=csv_dir,
                noiseflag=not args.no_noise,
            )
            summary[tag] = {
                "reward_mean":      m_r,
                "reward_std":       s_r,
                "shortfall_mean":   m_s,
                "shortfall_std":    s_s,
                "Reward":           r,
                "Action":           a,
            }
            logging.info("[DONE] %-25s  IS_mean=$%s", tag, f"{m_s:,.0f}")

    # ------------------------------------------------------------
    # Save summary CSV (tidy format)
    # ------------------------------------------------------------
    df_all = pd.DataFrame(summary).T.reset_index()
    df_all[["Reward", "Action"]] = df_all["index"].str.split("|", expand=True)
    df_all.drop(columns=["index"], inplace=True)

    # Reorder columns
    cols = ["Reward", "Action", "reward_mean", "reward_std", "shortfall_mean", "shortfall_std"]
    df_all = df_all[cols]

    csv_dir.mkdir(parents=True, exist_ok=True)
    df_all.to_csv(csv_dir / "summary_statistics.csv", index=False)

    # ------------------------------------------------------------
    # Optional plotting
    # ------------------------------------------------------------
    if args.plot:
        # Plot by reward (if >1 reward)
        if df_all["Reward"].nunique() > 1:
            plt.figure(figsize=(10, 6))
            sns.barplot(data=df_all, x="Reward", y="shortfall_mean",
                        palette="viridis", errorbar="sd")
            plt.ylabel("Avg Implementation Shortfall ($)")
            plt.xticks(rotation=45)
            plt.title(f"Shortfall by Reward  (Action = {df_all['Action'].iloc[0]})")
            plt.tight_layout()
            plt.savefig(csv_dir / "shortfall_by_reward.png")
            plt.close()

        # Plot by action (if >1 action)
        if df_all["Action"].nunique() > 1:
            plt.figure(figsize=(max(10, 0.9 * df_all['Action'].nunique()), 6))
            sns.barplot(data=df_all, x="Action", y="shortfall_mean",
                        palette="magma", errorbar="sd")
            plt.ylabel("Avg Implementation Shortfall ($)")
            plt.title(f"Shortfall by Action  (Reward = {df_all['Reward'].iloc[0]})")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.savefig(csv_dir / "shortfall_by_action.png")
            plt.close()



if __name__ == "__main__":
    main()
