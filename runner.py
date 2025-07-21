import seaborn as sns
import matplotlib.pyplot as plt
import inspect
import importlib
import gym
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import argparse
import numpy as np
import logging
from tqdm import trange
import pandas as pd
import os

from ddpg_agent import Agent as DDPGAgent
from actions import transform_action, TRANSFORMS
from rewards import REWARD_FN_MAP

# Import agents if available
try:
    from sac import SB3SACAgent
    from td3 import TD3Agent
except ImportError:
    SB3SACAgent = None
    TD3Agent = None


# Import utility functions
from utils import (
    plot_training_performance,
    plot_training_losses,
    plot_trade_list,
    plot_volatility_path,
    plot_price_path,
    plot_fee_impact
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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


def make_agent(
    agent_name: str, 
    env: gym.Env, 
    seed: int, 
    state_size: Optional[int] = None,
    action_size: Optional[int] = None,
    **kwargs
):
    """Factory function to create agents with support for Heston-Merton state size."""
    agent_name_lower = agent_name.lower()
    
    # Determine state and action sizes
    if state_size is None or action_size is None:
        try:
            state_size = env.observation_space.shape[0]
            action_size = env.action_space.shape[0]
        except AttributeError:
            # Use custom methods for Heston-Merton environment
            if hasattr(env, 'observation_space_dimension'):
                state_size = env.observation_space_dimension()
            if hasattr(env, 'action_space_dimension'):
                action_size = env.action_space_dimension()
    
    if agent_name_lower == "ddpg":
        return DDPGAgent(
            state_size=state_size, 
            action_size=action_size, 
            random_seed=seed,
            **kwargs
        )
    elif agent_name_lower == "sac" and SB3SACAgent is not None:
        return SB3SACAgent(env=env, seed=seed, **kwargs)
    elif agent_name_lower == "td3" and TD3Agent is not None:
        return TD3Agent(
            state_size=state_size, 
            action_size=action_size, 
            random_seed=seed, 
            env=env, 
            **kwargs
        )
    else:
        raise ValueError(f"Unknown agent '{agent_name}'. Use 'ddpg', 'sac', or 'td3'.")

def train_once(
    env_name: str, 
    agent_name: str, 
    reward_fn: str, 
    act_method: str, 
    episodes: int, 
    seed: int,
    csv_dir: Path, 
    noiseflag: bool = True, 
    fee_config: dict = None,
    agent_kwargs: dict = None,
    env_kwargs: dict = None
) -> Tuple[float, float, float, float, list, Any]:
    """Main training function with enhancements for Heston-Merton with fees."""
    log = logging.getLogger(f"{agent_name}|{reward_fn}|{act_method}")
    if agent_kwargs is None:
        agent_kwargs = {}
    if env_kwargs is None:
        env_kwargs = {}
    
    # Create environment with potential Heston-Merton parameters
    env = make_env(env_name, reward_fn, seed, fee_config, **env_kwargs)
    
    # Create agent - handle different state size for Heston-Merton
    state_size = None
    if hasattr(env, 'observation_space_dimension'):
        state_size = env.observation_space_dimension()
    agent = make_agent(
        agent_name, 
        env, 
        seed, 
        state_size=state_size,
        action_size=1,  # All our environments have 1D actions
        **agent_kwargs
    )
    
    rewards, shortfalls, utilities, fee_data, vol_paths, price_paths = [], [], [], [], [], []

    # For DDPG-style agents
    if agent_name.lower() == "ddpg":
        log.info(f"Starting DDPG training for {episodes} episodes...")
        for ep in trange(episodes, desc=f"Training {agent_name}"):
            state = env.reset(seed=seed + ep)
            agent.reset()
            done, tot_r = False, 0.0
            ep_trades = []  # Track trades in this episode
            ep_vol = []     # Track volatility path
            ep_prices = []  # Track price path
            
            # Start transactions explicitly for Heston-Merton
            if hasattr(env, 'start_transactions'):
                env.start_transactions()
            
            while not done:
                raw_action = agent.act(state, add_noise=noiseflag)
                action = transform_action(raw_action, env, act_method)
                next_state, reward_arr, done, info = env.step(action)
                reward_val = reward_arr[0]  # Unpack scalar from array
                
                agent.step(state, action, reward_val, next_state, done)
                tot_r += reward_val
                state = next_state
                
                # Record trade details
                if hasattr(info, 'share_to_sell_now'):
                    ep_trades.append(info.share_to_sell_now)
                
                # Record volatility and price
                if hasattr(env, 'current_variance'):
                    ep_vol.append(np.sqrt(env.current_variance))
                if hasattr(info, 'price'):
                    ep_prices.append(info.price)
            
            rewards.append(tot_r)
            
            # Handle completion info
            if hasattr(info, 'implementation_shortfall'):
                shortfalls.append(info.implementation_shortfall)
            if hasattr(info, 'utility'):
                utilities.append(info.utility)
            if hasattr(info, 'total_fees'):
                fee_data.append(info.total_fees)
            
            # Record paths for visualization
            vol_paths.append(ep_vol)
            price_paths.append(ep_prices)
            
            # Save trade list for analysis
            trade_df = pd.DataFrame({
                'step': range(len(ep_trades)),
                'shares': ep_trades
            })
            trade_path = csv_dir / f"trades_ep{ep}_{agent_name}_{reward_fn}_{act_method}.csv"
            trade_df.to_csv(trade_path, index=False)
            
            # Visualize first episode
            if ep == 0:
                # Plot trades
                plot_trade_list(
                    ep_trades, 
                    optimal_trades=env.get_trade_list() if hasattr(env, 'get_trade_list') else None,
                    file_path=csv_dir / f"{agent_name}_{reward_fn}_{act_method}_ep0_trades.png"
                )
                
                # Plot volatility path
                if ep_vol:
                    plot_volatility_path(
                        ep_vol,
                        file_path=csv_dir / f"{agent_name}_{reward_fn}_{act_method}_ep0_volatility.png"
                    )
                
                # Plot price path
                if ep_prices:
                    plot_price_path(
                        ep_prices,
                        trades=ep_trades,
                        file_path=csv_dir / f"{agent_name}_{reward_fn}_{act_method}_ep0_price.png"
                    )

    # For SAC/TD3 agents
    elif agent_name.lower() in ["sac", "td3"]:
        # Calculate total timesteps (number of trades per episode times episodes)
        timesteps_per_episode = env.num_n if hasattr(env, 'num_n') else 100
        total_timesteps = episodes * timesteps_per_episode
        log.info(f"Starting {agent_name} training for {total_timesteps} timesteps...")
        agent.learn(total_timesteps=total_timesteps)
        
        log.info(f"Evaluating {agent_name} for {episodes} episodes...")
        for ep in trange(episodes, desc=f"Evaluating {agent_name}"):
            state = env.reset(seed=seed + ep)
            done, tot_r = False, 0.0
            ep_trades = []
            ep_vol = []
            ep_prices = []
            
            if hasattr(env, 'start_transactions'):
                env.start_transactions()
                
            while not done:
                raw_action = agent.act(state, deterministic=True)
                action = transform_action(raw_action, env, act_method)
                state, reward_arr, done, info = env.step(action)
                reward_val = reward_arr[0]
                tot_r += reward_val
                
                if hasattr(info, 'share_to_sell_now'):
                    ep_trades.append(info.share_to_sell_now)
                if hasattr(env, 'current_variance'):
                    ep_vol.append(np.sqrt(env.current_variance))
                if hasattr(info, 'price'):
                    ep_prices.append(info.price)
            
            rewards.append(tot_r)
            if hasattr(info, 'implementation_shortfall'):
                shortfalls.append(info.implementation_shortfall)
            if hasattr(info, 'utility'):
                utilities.append(info.utility)
            if hasattr(info, 'total_fees'):
                fee_data.append(info.total_fees)
            
            vol_paths.append(ep_vol)
            price_paths.append(ep_prices)
            
            if ep == 0:
                plot_trade_list(
                    ep_trades, 
                    optimal_trades=env.get_trade_list() if hasattr(env, 'get_trade_list') else None,
                    file_path=csv_dir / f"{agent_name}_{reward_fn}_{act_method}_ep0_trades.png"
                )
                if ep_vol:
                    plot_volatility_path(
                        ep_vol,
                        file_path=csv_dir / f"{agent_name}_{reward_fn}_{act_method}_ep0_volatility.png"
                    )
                if ep_prices:
                    plot_price_path(
                        ep_prices,
                        trades=ep_trades,
                        file_path=csv_dir / f"{agent_name}_{reward_fn}_{act_method}_ep0_price.png"
                    )

    # Save results
    csv_dir.mkdir(parents=True, exist_ok=True)
    file_tag = f"{agent_name}_{reward_fn}_{act_method}"
    
    # Prepare results dataframe
    results = {
        "episode": range(len(rewards)),
        "reward": rewards,
    }
    if shortfalls:
        results["shortfall"] = shortfalls
    if utilities:
        results["utility"] = utilities
    if fee_data:
        results["total_fees"] = fee_data
        
    results_df = pd.DataFrame(results)
    results_df.to_csv(csv_dir / f"{file_tag}.csv", index=False)
    
    # Save paths for later analysis
    if vol_paths:
        vol_df = pd.DataFrame(vol_paths).T
        vol_df.to_csv(csv_dir / f"{file_tag}_volatility_paths.csv", index=False)
    
    if price_paths:
        price_df = pd.DataFrame(price_paths).T
        price_df.to_csv(csv_dir / f"{file_tag}_price_paths.csv", index=False)
    
    # Generate plots
    plot_file_path = csv_dir / f"{file_tag}_performance.png"
    plot_training_performance(rewards, shortfalls, utilities, fee_data, window_size=100, file_path=plot_file_path)

    # Plot fee impact analysis
    if fee_data:
        plot_fee_impact(
            shortfalls, 
            fee_data,
            file_path=csv_dir / f"{file_tag}_fee_impact.png"
        )
    
    # Plot losses for DDPG
    if agent_name.lower() == "ddpg":
        loss_file_path = csv_dir / f"{file_tag}_losses.png"
        plot_training_losses(agent, window_size=100, file_path=loss_file_path)
    
    return (
        np.mean(rewards) if rewards else 0,
        np.std(rewards) if rewards else 0,
        np.mean(shortfalls) if shortfalls else 0,
        np.std(shortfalls) if shortfalls else 0,
        shortfalls,
        agent
    )

def main(argv: List[str] | None = None):
    p = argparse.ArgumentParser("Heston-Merton Trading Runner")
    p.add_argument("--agent", default="ddpg", help="Agent name: 'ddpg', 'sac', or 'td3'")
    p.add_argument("--env", default="heston_merton_fees", help="Environment name")
    p.add_argument("--reward", default="ac_utility", help="Reward function")
    p.add_argument("--action", default="linear", help="Action transform method")
    p.add_argument("--episodes", type=int, default=1000, help="Number of training episodes")
    p.add_argument("--seed", type=int, default=0, help="Random seed")
    p.add_argument("--csv-dir", default="results/ChrissAlmgren", help="Output directory for results")
    p.add_argument("--no-noise", action="store_true", help="Disable exploration noise")
    p.add_argument("--plot", action="store_true", help="Generate plots")
    
    # Heston-Merton specific parameters
    p.add_argument("--total-shares", type=int, default=1000000, help="Total shares to liquidate")
    p.add_argument("--liquidation-time", type=int, default=60, help="Liquidation time horizon (days)")
    p.add_argument("--risk-aversion", type=float, default=1e-6, dest="lambda", help="Risk aversion parameter")
    p.add_argument("--starting-price", type=float, default=50.0, help="Initial stock price")
    p.add_argument("--annual-volat", type=float, default=0.12, help="Annual volatility")
    
    # Heston parameters
    p.add_argument("--heston-kappa", type=float, default=3.0, help="Volatility mean-reversion speed")
    p.add_argument("--heston-theta", type=float, default=0.0144, help="Long-term variance (0.12^2=0.0144)")
    p.add_argument("--heston-sigma-v", type=float, default=0.1, help="Volatility of volatility")
    p.add_argument("--heston-rho", type=float, default=-0.7, help="Price-vol correlation")
    p.add_argument("--heston-v0", type=float, default=0.0144, help="Initial variance")
    
    # Merton parameters
    p.add_argument("--jump-lambda", type=float, default=0.5, help="Jump intensity (jumps/year)")
    p.add_argument("--jump-mu", type=float, default=-0.05, help="Mean jump size (log)")
    p.add_argument("--jump-sigma", type=float, default=0.1, help="Jump size volatility")
    
    # Fee parameters
    p.add_argument("--fee-fixed", type=float, default=10.0, help="Fixed fee per trade")
    p.add_argument("--fee-prop", type=float, default=0.001, help="Proportional fee rate")
    
    # Agent-specific parameters
    p.add_argument("--actor-lr", type=float, default=1e-4, help="Actor learning rate")
    p.add_argument("--critic-lr", type=float, default=1e-3, help="Critic learning rate")
    p.add_argument("--tau", type=float, default=1e-3, help="Soft update parameter")
    p.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    
    args = p.parse_args(argv)
    
    # Create output directory
    csv_dir = Path(args.csv_dir)
    csv_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config = vars(args)
    pd.DataFrame([config]).to_csv(csv_dir / "config.csv", index=False)
    
    # Prepare fee config
    fee_config = {
        "fixed": args.fee_fixed,
        "prop": args.fee_prop
    }
    
    # Prepare environment parameters
    env_kwargs = {
        "total_shares": args.total_shares,
        "liquidation_time": args.liquidation_time,
        "startingPrice": args.starting_price,
        "anv": args.annual_volat,
        "heston_kappa": args.heston_kappa,
        "heston_theta": args.heston_theta,
        "heston_sigma_v": args.heston_sigma_v,
        "heston_rho": args.heston_rho,
        "heston_v0": args.heston_v0,
        "jump_lambda": args.jump_lambda,
        "jump_mu": args.jump_mu,
        "jump_sigma": args.jump_sigma,
        "llambda": args.llambda
    }
    
    # Prepare agent parameters
    agent_kwargs = {
        "actor_lr": args.actor_lr,
        "critic_lr": args.critic_lr,
        "tau": args.tau,
        "gamma": args.gamma
    }
    
    # Run training
    logger.info("Starting training with configuration:")
    for k, v in config.items():
        logger.info(f"{k:>20}: {v}")
    
    (mean_reward, std_reward, 
     mean_shortfall, std_shortfall,
     shortfall_history, agent) = train_once(
        env_name=args.env,
        agent_name=args.agent,
        reward_fn=args.reward,
        act_method=args.action,
        episodes=args.episodes,
        seed=args.seed,
        csv_dir=csv_dir,
        noiseflag=not args.no_noise,
        fee_config=fee_config,
        agent_kwargs=agent_kwargs,
        env_kwargs=env_kwargs
    )
    
    logger.info(f"Training completed. Mean shortfall: ${mean_shortfall:,.2f} ± ${std_shortfall:,.2f}")
    
    # Save final agent
    if hasattr(agent, 'save'):
        agent.save(csv_dir / f"{args.agent}_final.pth")
    
    # Generate summary report
    summary = {
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "mean_shortfall": mean_shortfall,
        "std_shortfall": std_shortfall,
        "agent": args.agent,
        "env": args.env,
        "reward_fn": args.reward,
        "action_transform": args.action
    }
    pd.DataFrame([summary]).to_csv(csv_dir / "summary.csv", index=False)

if __name__ == "__main__":
    main()