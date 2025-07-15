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
import inspect
import gym

from rewards import REWARD_FN_MAP

from syntheticChrissAlmgren_extended import MarketEnvironment as AlmgrenChrisEnvironment
from ddpg_agent import Agent as DDPGAgent
from actions import transform_action, TRANSFORMS

from sac import SB3SACAgent


def make_env(env_name: str, seed: int, fee_config: dict = None) -> gym.Env:
    """
    Creates an environment instance based on its name.
    This function handles multiple environment types and fee configurations.
    """
    env_name_lower = env_name.lower()
    if env_name_lower == "ac_default":
        return AlmgrenChrisEnvironment(randomSeed=seed)
    

    try:
        if env_name_lower == "gbm":
            from GBM import GBMMarketEnvironment
            return GBMMarketEnvironment(randomSeed=seed)
        elif env_name_lower == "heston_merton":
            from Hetson_Merton_Env import HestonMertonEnvironment
            return HestonMertonEnvironment(randomSeed=seed, fee_config=fee_config)
        elif env_name_lower == "heston_merton_fees":
            from Hetson_Merton_fees import HestonMertonFeesEnvironment
            return HestonMertonFeesEnvironment(randomSeed=seed, fee_config=fee_config)
        else:
            # Fallback for custom 'module:ClassName' syntax
            # Dynamic import for custom environments
            module_name, cls_name = env_name.split(":")
            mod = importlib.import_module(module_name)
            cls = getattr(mod, cls_name)
            try:
                return cls(randomSeed=seed, fee_config=fee_config)
            except TypeError:
                return cls(randomSeed=seed)
    except (ImportError, ModuleNotFoundError, ValueError, AttributeError) as e:
        raise ValueError(f"Could not create environment '{env_name}'. Error: {e}")      


def make_agent(agent_name: str, env: gym.Env, seed: int, **kwargs):
    """
    Factory function to create DDPG or SAC agents.
    It takes the 'env' object directly for compatibility and passes hyperparameters via kwargs.
    """
    # The environment must have valid observation and action spaces.
    if not hasattr(env, 'observation_space') or not hasattr(env, 'action_space'):
        raise TypeError("The provided environment is not a valid Gym environment.")

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent_name_lower = agent_name.lower()

    if agent_name_lower == "ddpg":
        return DDPGAgent(state_size=state_dim, action_size=action_dim, random_seed=seed)
    
    elif agent_name_lower == "sac":
        return SB3SACAgent(env=env, seed=seed, **kwargs)
        
    else:
        raise ValueError(f"Unknown agent '{agent_name}'. Use 'ddpg' or 'sac'.")


def supports_reward_function(env_step_method):
    """Check if environment's step method supports reward_function parameter."""
    sig = inspect.signature(env_step_method)
    return 'reward_function' in sig.parameters


def train_once(env_name: str, agent_name: str, reward_fn: str, act_method: str, episodes: int, seed: int,
               csv_dir: Path, noiseflag: bool = True, fee_config: dict = None, agent_kwargs: dict = None):
    """
    Main training and evaluation function with separate logic for DDPG and SAC.
    """
    log = logging.getLogger(f"{agent_name}|{reward_fn}|{act_method}")
    if agent_kwargs is None:
        agent_kwargs = {}

    env = make_env(env_name, seed, fee_config)
    agent = make_agent(agent_name, env, seed, **agent_kwargs)
    
    step_supports_reward = supports_reward_function(env.step)
    rewards, shortfalls, fee_data = [], [], []

    if agent_name.lower() == "sac":
        total_timesteps = episodes * env.num_n
        log.info(f"Starting SAC training for {total_timesteps} timesteps...")
        agent.learn(total_timesteps=total_timesteps)
        log.info("SAC training finished.")

        log.info(f"Starting SAC evaluation for {episodes} episodes...")
        for ep in trange(episodes, desc=f"Evaluating {agent_name}"):
            state = env.reset(seed=seed + ep)
            env.start_transactions()
            done, tot_r, episode_fees = False, 0.0, 0.0
            while not done:
                raw_action = agent.act(state, deterministic=True)
                action = transform_action(raw_action, env, act_method)
                step_args = (action, reward_fn) if step_supports_reward else (action,)
                next_state, reward, done, info = env.step(*step_args)
                tot_r += reward
                state = next_state
                if hasattr(info, 'total_fees'):
                    episode_fees += info.total_fees
            rewards.append(tot_r)
            shortfalls.append(info.implementation_shortfall)
            fee_data.append(episode_fees)
    else:
        log.info(f"Starting DDPG training for {episodes} episodes...")
        for ep in trange(episodes, desc=f"Training {agent_name}"):
            state = env.reset(seed=seed + ep)
            env.start_transactions()
            agent.reset()
            done, tot_r, episode_fees = False, 0.0, 0.0
            while not done:
                raw_action = agent.act(state, add_noise=noiseflag)
                action = transform_action(raw_action, env, act_method)
                step_args = (action, reward_fn) if step_supports_reward else (action,)
                next_state, reward, done, info = env.step(*step_args)
                r_scalar = reward.item() if hasattr(reward, 'item') else reward
                agent.step(state, action, r_scalar, next_state, done)
                tot_r += r_scalar
                state = next_state
                if hasattr(info, 'total_fees'):
                    episode_fees += info.total_fees
            rewards.append(tot_r)
            shortfalls.append(info.implementation_shortfall)
            fee_data.append(episode_fees)

    csv_dir.mkdir(parents=True, exist_ok=True)
    df_data = {"episode": range(len(rewards)), "reward": rewards, "shortfall": shortfalls}
    if any(fee > 0 for fee in fee_data):
        df_data["total_fees"] = fee_data
    file_tag = f"{agent_name}_{reward_fn}_{act_method}"
    pd.DataFrame(df_data).to_csv(csv_dir / f"{file_tag}.csv", index=False)
    return (np.mean(rewards), np.std(rewards), np.mean(shortfalls), np.std(shortfalls))


def run_action_transform_test(transform_methods: List[str], seed_count: int, output_dir: Path, env_name: str = "ac_default"): 
    """Test action transformation methods with dynamic environment support."""
    results = {}
    for method in transform_methods:
        shortfalls = []
        for seed in range(seed_count):
            env = make_env(env_name, seed)
            env.start_transactions()
            agent = DDPGAgent(state_size=env.observation_space.shape[0], action_size=1, random_seed=seed)
            state = env.reset(seed=seed)
            done = False
            while not done:
                raw_action = agent.act(state, add_noise=False)
                action = transform_action(raw_action, env, method)
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
    p.add_argument("--env", default="ac_default", 
                   help="Environment: 'ac_default', 'gbm', 'heston_merton', 'heston_merton_fees', or 'module:Class'")
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
    
    # Fee configuration arguments
    p.add_argument("--fee-fixed", type=float, default=10.0, help="Fixed fee per trade")
    p.add_argument("--fee-prop", type=float, default=0.001, help="Proportional fee rate")
    
    # Action transform test arguments
    p.add_argument("--run-action-test", action="store_true", help="Run action transformation test")
    p.add_argument("--test-seeds", type=int, default=10, help="Number of seeds for action transform test")

    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    csv_dir = Path(args.csv_dir)

    # Determine which agents, rewards, and actions to run
    agents_to_run = ['ddpg', 'sac'] if args.agent.lower() == 'all' else [args.agent]
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

    
    # Run action transformation test if requested
    if args.run_action_test:
        run_action_transform_test(args.transform_methods, args.test_seeds, csv_dir, args.env)
        return

    
    # Configure fees
    fee_config = {
        "fixed": args.fee_fixed,
        "prop": args.fee_prop
    }

    summary: dict[str, dict[str, float]] = {}

    for agent_to_run in agents_to_run:
        for r in reward_batch:
            for a in action_batch:
                tag = f"{agent_to_run}|{r}|{a}"
                logging.info(f"----- Running experiment for: {tag} -----")

                agent_kwargs = {}
                if agent_to_run == "sac":
                    agent_kwargs["policy_kwargs"] = dict(net_arch=[256, 256])
                    agent_kwargs["learning_rate"] = 0.0003
                    # more SAC parameters here if needed

                m_r, s_r, m_s, s_s = train_once(
                    env_name=args.env,
                    agent_name=agent_to_run,
                    reward_fn=r,
                    act_method=a,
                    episodes=args.episodes,
                    seed=args.seed,
                    csv_dir=csv_dir,
                    noiseflag=not args.no_noise,
                    fee_config=fee_config,
                    agent_kwargs=agent_kwargs
                )
                summary[tag] = {
                    "agent": agent_to_run,
                    "reward_fn": r,
                    "action_fn": a,
                    "reward_mean": m_r,
                    "reward_std": s_r,
                    "shortfall_mean": m_s,
                    "shortfall_std": s_s,
                }
                logging.info("[DONE] %-35s | Avg Shortfall: $%s", tag, f"{m_s:,.0f}")

    if not summary:
        logging.warning("No experiments were run. Exiting.")
        return

    # ------------------------------------------------------------
    # Save summary CSV (tidy format)
    # ------------------------------------------------------------
    df_all = pd.DataFrame(summary).T.reset_index()
    df_all.rename(columns={
    "agent": "Agent",
    "reward_fn": "Reward",
    "action_fn": "Action"}, inplace=True)
    
    df_all = df_all[["Agent", "Reward", "Action", "reward_mean", "reward_std", "shortfall_mean", "shortfall_std"]]

    # Reorder columns
    cols = ["Reward", "Action", "reward_mean", "reward_std", "shortfall_mean", "shortfall_std"]
    df_all = df_all[cols]

    csv_dir.mkdir(parents=True, exist_ok=True)
    df_all.to_csv(csv_dir / "summary_statistics.csv", index=False)

    # ------------------------------------------------------------
    # Optional plotting
    # ------------------------------------------------------------
    if args.plot:
        if df_all["Reward"].nunique() > 1:
            plt.figure(figsize=(10, 6))
            sns.barplot(data=df_all, x="Reward", y="shortfall_mean", hue="Agent",
                        palette="viridis", errorbar="sd", dodge=True)
            plt.ylabel("Avg Implementation Shortfall ($)")
            plt.title("Shortfall by Reward Function")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(csv_dir / "shortfall_by_reward.png")
            plt.close()

        if df_all["Action"].nunique() > 1:
            plt.figure(figsize=(max(10, 0.9 * df_all['Action'].nunique()), 6))
            sns.barplot(data=df_all, x="Action", y="shortfall_mean", hue="Agent",
                        palette="magma", errorbar="sd", dodge=True)
            plt.ylabel("Avg Implementation Shortfall ($)")
            plt.title("Shortfall by Action Transform")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.savefig(csv_dir / "shortfall_by_action.png")
            plt.close()


if __name__ == "__main__":
    main()