import seaborn as sns
import matplotlib.pyplot as plt
import inspect
import gym
from pathlib import Path
from typing import List
import argparse
import numpy as np
import logging
from tqdm import trange
import pandas as pd

from ddpg_agent import Agent as DDPGAgent
from actions import transform_action, TRANSFORMS
from rewards import REWARD_FN_MAP

from sac import SB3SACAgent
from td3 import TD3Agent

from utils import plot_training_performance, plot_training_losses


def make_env(env_name: str, reward_fn: str, seed: int, fee_config: dict = None) -> gym.Env:
    """
    Creates an environment instance.
    """
    if env_name == "ac_default":
        from syntheticChrissAlmgren import MarketEnvironment
        return MarketEnvironment(randomSeed=seed, reward_fn=reward_fn)
    elif env_name == "gbm":
        from GBM import GBMMarketEnvironment
        return GBMMarketEnvironment(randomSeed=seed, reward_fn=reward_fn)
    elif env_name == "heston_merton":
        from Hetson_Merton_Env import HestonMertonEnvironment
        return HestonMertonEnvironment(randomSeed=seed, reward_fn=reward_fn)
    elif env_name == "heston_merton_fees":
        from Hetson_Merton_fees import HestonMertonFeesEnvironment
        return HestonMertonEnvironment(randomSeed=seed, fee_config=fee_config, reward_fn=reward_fn)
    else:
        try:
            module_name, cls_name = env_name.split(":")
            mod = importlib.import_module(module_name)
            cls = getattr(mod, cls_name)
            try:
                return cls(randomSeed=seed, fee_config=fee_config)
            except TypeError:
                return cls(randomSeed=seed)
        except (ValueError, ModuleNotFoundError, AttributeError):
            raise ValueError(f"Unknown env format '{env_name}'.")    


def make_agent(agent_name: str, env: gym.Env, seed: int, **kwargs):
    """
    Factory function to create DDPG, SAC, or TD3 agents.
    """
    agent_name_lower = agent_name.lower()

    # Handle custom environment interface
    try:
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.shape[0]
    except AttributeError:
        # Fallback to custom methods
        state_size = env.observation_space_dimension()
        action_size = env.action_space_dimension()

    if agent_name_lower == "ddpg":
        return DDPGAgent(state_size=state_size, action_size=action_size, random_seed=seed)
    elif agent_name_lower == "sac":
        return SB3SACAgent(env=env, seed=seed, **kwargs)
    elif agent_name_lower == "td3":
        return TD3Agent(state_size=state_size, action_size=action_size, random_seed=seed, env=env, **kwargs)
    else:
        raise ValueError(f"Unknown agent '{agent_name}'. Use 'ddpg', 'sac', or 'td3'.")


def supports_reward_function(env_step_method):
    """Check if environment's step method supports reward_function parameter."""
    sig = inspect.signature(env_step_method)
    return 'reward_function' in sig.parameters


def train_once(env_name: str, agent_name: str, reward_fn: str, act_method: str, episodes: int, seed: int,
               csv_dir: Path, noiseflag: bool = True, fee_config: dict = None, agent_kwargs: dict = None):
    """
    Main training function with separate logic for DDPG and SB3 agents.
    Returns tuple of (mean_reward, std_reward, mean_shortfall, std_shortfall, shortfall_history, agent)
    """
    log = logging.getLogger(f"{agent_name}|{reward_fn}|{act_method}")
    if agent_kwargs is None:
        agent_kwargs = {}

    env = make_env(env_name, reward_fn, seed, fee_config)
    agent = make_agent(agent_name, env, seed, **agent_kwargs)
    
    rewards, shortfalls, fee_data = [], [], []

    if agent_name.lower() in ["sac" , "td3"]:
        # --- SB3 (SAC) Workflow: Learn then Evaluate ---
        total_timesteps = episodes * env.num_n
        log.info(f"Starting SAC training for {total_timesteps} timesteps...")
        agent.learn(total_timesteps=total_timesteps)
        log.info("SAC training finished.")

        log.info(f"Starting SAC evaluation for {episodes} episodes...")
        for ep in trange(episodes, desc=f"Evaluating {agent_name}"):
            state, _ = env.reset(seed=seed + ep)
            done, tot_r, episode_fees = False, 0.0, 0.0
            while not (done):
                raw_action = agent.act(state, deterministic=True)
                action = transform_action(raw_action, env, act_method)
                state, reward, done, info = env.step(action)
                tot_r += reward

            rewards.append(tot_r)
            shortfalls.append(info.get("impl_shortfall", np.nan))
            fee_data.append(episode_fees)
            
    else: # DDPG Workflow
        log.info(f"Starting DDPG training for {episodes} episodes...")
        for ep in trange(episodes, desc=f"Training {agent_name}"):
            state = env.reset(seed=seed + ep)
            agent.reset()
            done, tot_r, episode_fees = False, 0.0, 0.0
            while not (done):
                raw_action = agent.act(state, add_noise=noiseflag)
                action = transform_action(raw_action, env, act_method)
                next_state, reward, done, info = env.step(action)
                agent.step(state, action, reward, next_state, done)
                tot_r += reward
                state = next_state
                
            rewards.append(tot_r)
            shortfalls.append(info.get("impl_shortfall", np.nan))
            fee_data.append(episode_fees)

    # --- Save Results ---
    csv_dir.mkdir(parents=True, exist_ok=True)
    df_data = {"episode": range(len(rewards)), "reward": rewards, "shortfall": shortfalls}
    if any(fee > 0 for fee in fee_data):
        df_data["total_fees"] = fee_data
    file_tag = f"{agent_name}_{reward_fn}_{act_method}"
    pd.DataFrame(df_data).to_csv(csv_dir / f"{file_tag}.csv", index=False)
    
    # Return metrics plus the shortfall history and agent instance for plotting
    return (
        np.mean(rewards), np.std(rewards), 
        np.mean(shortfalls), np.std(shortfalls),
        shortfalls, agent
    )


def run_action_transform_test(transform_methods: List[str], seed_count: int, output_dir: Path, env_name: str):
    """Dedicated test for action transformation methods."""
    results = {}
    for method in transform_methods:
        shortfalls = []
        for seed in range(seed_count):
            env = make_env(env_name, "ac_utility", seed)
            agent = DDPGAgent(state_size=env.observation_space.shape[0], action_size=1, random_seed=seed)
            state, _ = env.reset(seed=seed)
            done= False
            while not done:
                raw_action = agent.act(state, add_noise=False)
                action = transform_action(raw_action, env._ac_env, method)
                state, _, done, info = env.step(action)
            shortfalls.append(info.get("impl_shortfall", np.nan))
        results[method] = {'mean_shortfall': np.nanmean(shortfalls), 'std_shortfall': np.nanstd(shortfalls)}
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
    p.add_argument("--compare-agents", action="store_true", help="Batch-compare all agents (ddpg, sac, td3).")
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
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    csv_dir = Path(args.csv_dir)
    fee_config = {"fixed": args.fee_fixed, "prop": args.fee_prop}

    # --- Action Test Mode ---
    if args.run_action_test:
        logging.info("Running dedicated action transform test...")
        run_action_transform_test(args.transform_methods, args.test_seeds, csv_dir, args.env)
        return

    # --- Batch and Agent Selection (Updated Logic) ---
    # Use the new --compare-agents flag for selecting agents
    agents_to_run = ['ddpg', 'sac', 'td3'] if args.compare_agents else [args.agent]
    
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

    summary: dict[str, dict] = {}

    for agent_to_run in agents_to_run:
        for r in reward_batch:
            for a in action_batch:
                tag = f"{agent_to_run}|{r}|{a}"
                logging.info(f"----- Running experiment for: {tag} -----")
                
                agent_kwargs = {}
                if agent_to_run == "sac":
                    agent_kwargs['policy_kwargs'] = dict(net_arch=[256, 256])
                    agent_kwargs['learning_rate'] = 0.0003
                elif agent_to_run == "td3":
                    agent_kwargs['policy_kwargs'] = dict(net_arch=[256, 256])
                    agent_kwargs['learning_rate'] = 0.001
                    agent_kwargs['policy_delay'] = 2
                    agent_kwargs['target_policy_noise'] = 0.2
                    agent_kwargs['target_noise_clip'] = 0.5
                    agent_kwargs['batch_size'] = 256
                
                m_r, s_r, m_s, s_s, shortfall_history, trained_agent = train_once(
                    env_name=args.env, agent_name=agent_to_run,
                    reward_fn=r, act_method=a,
                    episodes=args.episodes, seed=args.seed,
                    csv_dir=csv_dir, noiseflag=not args.no_noise,
                    fee_config=fee_config, agent_kwargs=agent_kwargs
                )
                # Generate plots after training
                plot_file_path = csv_dir / f"{tag}_training_performance.png"
                plot_training_performance(shortfall_history, window_size=100, file_path=plot_file_path)


                # Only plot losses for DDPG agent (SAC/TD3 would need different handling)
                if agent_to_run.lower() == "ddpg":
                    loss_file_path = csv_dir / f"{tag}_training_losses.png"
                    plot_training_losses(trained_agent, window_size=100, file_path=loss_file_path)

                
                summary[tag] = {
                    "Agent": agent_to_run, "Reward": r, "Action": a,
                    "reward_mean": m_r, "reward_std": s_r,
                    "shortfall_mean": m_s, "shortfall_std": s_s,
                }
                logging.info("[DONE] %-35s | Avg Shortfall: $%s", tag, f"{m_s:,.0f} ± ${s_s:,.0f}")

    if not summary:
        logging.warning("No experiments were run. Exiting.")
        return
        
    # --- CSV Saving ---
    df_all = pd.DataFrame(summary).T
    cols = ["Agent", "Reward", "Action", "reward_mean", "reward_std", "shortfall_mean", "shortfall_std"]
    df_all = df_all.reset_index(drop=True)[cols]
    summary_path = csv_dir / "summary_statistics.csv"
    df_all.to_csv(summary_path, index=False)
    logging.info(f"\nSummary statistics saved to {summary_path}")
    print("\n" + df_all.to_string())

    # --- Plotting (with new agent comparison plot) ---
    if args.plot and len(summary) > 1:
        if df_all["Agent"].nunique() > 1:
            logging.info("Generating agent comparison plot...")
            g = sns.catplot(
                data=df_all,
                x="Agent",
                y="shortfall_mean",
                col="Reward",
                row="Action",
                kind="bar",
                palette="viridis",
                sharey=False
            )
            g.fig.suptitle("Agent Comparison by Avg. Implementation Shortfall", y=1.03)
            g.set_axis_labels("Agent", "Avg Implementation Shortfall ($)")
            g.set_titles("Reward: {col_name} | Action: {row_name}")
            g.fig.tight_layout(rect=[0, 0, 1, 0.97])
            plt.savefig(csv_dir / "shortfall_by_agent.png")
            plt.close('all')

        if df_all["Reward"].nunique() > 1:
            plt.figure(figsize=(10, 6))
            sns.barplot(data=df_all, x="Reward", y="shortfall_mean", hue="Agent", palette="viridis", dodge=True)
            plt.ylabel("Avg Implementation Shortfall ($)")
            plt.title("Shortfall by Reward Function")
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(csv_dir / "shortfall_by_reward.png")
            plt.close()
            
        if df_all["Action"].nunique() > 1:
            plt.figure(figsize=(max(10, 0.9 * df_all['Action'].nunique()), 6))
            sns.barplot(data=df_all, x="Action", y="shortfall_mean", hue="Agent", palette="magma", dodge=True)
            plt.ylabel("Avg Implementation Shortfall ($)")
            plt.title("Shortfall by Action Transform")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.savefig(csv_dir / "shortfall_by_action.png")
            plt.close()
            

if __name__ == "__main__":
    main()