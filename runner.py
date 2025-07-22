# runner.py
import argparse
import numpy as np
from collections import deque
import torch
import matplotlib.pyplot as plt
from syntheticChrissAlmgren import MarketEnvironment
from GBM import GBMMarketEnvironment
from Hetson_Merton_Env import HMMarketEnvironment
from Hetson_Merton_fees import HMFMarketEnvironment
from ddpg_agent import Agent
from rewards import REWARD_FN_MAP
from actions import transform_action, TRANSFORMS
import os
import sys
import pandas as pd
from datetime import datetime

from utils import plot_training_losses
from utils import plot_training_performance

def parse_args():
    parser = argparse.ArgumentParser(description='DDPG Agent for Optimal Trade Execution')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--reward', type=str, default='ac_utility', choices=REWARD_FN_MAP.keys(), help='Reward function')
    parser.add_argument('--env', type=str, default="AC", choices=["AC", "GBM", "HM", "HMF"], help='Market environment type')
    parser.add_argument('--action', type=str, default="linear", choices=TRANSFORMS.keys(), help="agent action strartegy")
    # Fee arguments (only used with HMF environment)
    fee_group = parser.add_argument_group('HMF environment fees', 'Parameters specific to Heston-Merton with Fees environment')
    fee_group.add_argument('--fixed_fee', type=float, default=10.0,
                         help='Fixed fee per trade (only for HMF environment)')
    fee_group.add_argument('--proportional_fee', type=float, default=0.001,
                         help='Proportional fee rate (only for HMF environment)')

    parser.add_argument('--lqd_time', type=float, default=60.0, help='Liquidation time (days)')
    parser.add_argument('--num_tr', type=int, default=60, help='Number of trades')
    parser.add_argument('--lambd', type=float, default=1e-6, help='Risk aversion parameter')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of training episodes')

    args = parser.parse_args()

    # Validate that fee arguments are only provided with HMF environment
    if args.env != "HMF" and (args.fixed_fee != 10.0 or args.proportional_fee != 0.001):
        print("Warning: Fee parameters are only applicable to HMF environment and will be ignored", 
              file=sys.stderr)

    return args

def save_results(agent_name, env_name, reward_func, avg_implementation_shortfall,
                 fixed_fee=None, proportional_fee=None, filename='results/results.csv'):
    
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Create DataFrame with the results
    data = {
        'agent': [agent_name],
        'env': [env_name],
        'reward_func': [reward_func],
        'avg_implementation_shortfall': [avg_implementation_shortfall],
        'timestamp': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
    }

    # Add fee parameters if they exist
    if fixed_fee is not None:
        data['fixed_fee'] = [fixed_fee]
    if proportional_fee is not None:
        data['proportional_fee'] = [proportional_fee]

    df = pd.DataFrame(data)
    
    # Check if file exists to determine whether to write header
    file_exists = os.path.isfile(filename)
    
    # Save to CSV
    df.to_csv(filename, mode='a', header=not file_exists, index=False)

def main():
    args = parse_args()
    
    if args.env == "AC":
        # Initialize environment
        env = MarketEnvironment(
            randomSeed=args.seed,
            lqd_time=args.lqd_time,
            num_tr=args.num_tr,
            lambd=args.lambd,
            reward_fn=args.reward
        )

    elif args.env == "GBM":  
        # Initialize environment
        env = GBMMarketEnvironment(
            randomSeed=args.seed,
            lqd_time=args.lqd_time,
            num_tr=args.num_tr,
            lambd=args.lambd,
            reward_fn=args.reward
        )
        
    elif args.env == "HM":  
        # Initialize environment
        env = HMMarketEnvironment(
            randomSeed=args.seed,
            lqd_time=args.lqd_time,
            num_tr=args.num_tr,
            lambd=args.lambd,
            reward_fn=args.reward
        )

    elif args.env == "HMF":  
        # Initialize environment
        env = HMFMarketEnvironment(
            randomSeed=args.seed,
            lqd_time=args.lqd_time,
            num_tr=args.num_tr,
            lambd=args.lambd,
            reward_fn=args.reward,
            fixed_fee=args.fixed_fee,
            proportional_fee=args.proportional_fee
        )                               
    else:
        raise ValueError(f"Unknown environment type: {args.env}")


    # Initialize agent
    agent = Agent(
        state_size=env.observation_space_dimension(),
        action_size=env.action_space_dimension(),
        random_seed=args.seed
    )
    
    episodes = args.episodes
    lqt = args.lqd_time
    n_trades = args.num_tr
    tr = args.lambd

    shortfall_hist = np.array([])
    shortfall_deque = deque(maxlen=100)
    
    print(f"Starting training with {args.episodes} episodes...")

    print(f"Parameters: seed={args.seed}, reward={args.reward}, lqd_time={args.lqd_time}, "
          f"num_tr={args.num_tr}, lambd={args.lambd}")
    
    if args.env == "HMF":
        print(f"Fee parameters: fixed_fee={args.fixed_fee}, proportional_fee={args.proportional_fee}")

    for episode in range(episodes):
        # Reset environment with new seed for each episode
        cur_state = env.reset(seed = episode, reward_fn=args.reward, liquid_time = lqt, num_trades = n_trades, lamb = tr)

        # set the environment to make transactions
        env.start_transactions()
        for i in range(n_trades + 1):

            # Predict the best action for the current state. 
            raw_action = agent.act(cur_state, add_noise=True)
            action = transform_action(raw_action, env, args.action)
        
            # Action is performed and new state, reward, info are received. 
            new_state, reward, done, info = env.step(action)
        
            # current state, action, reward, new state are stored in the experience replay
            agent.step(cur_state, action, reward, new_state, done)
        
            # roll over new state
            cur_state = new_state

            if info.done:
                shortfall_hist = np.append(shortfall_hist, info.implementation_shortfall)
                shortfall_deque.append(info.implementation_shortfall)
                break
        
        if (episode + 1) % 100 == 0: # print average shortfall over last 100 episodes
            print('\rEpisode [{}/{}]\tAverage Shortfall: ${:,.2f}'.format(episode + 1, episodes, np.mean(shortfall_deque)))        

    avg_shortfall = np.mean(shortfall_hist)
    print('\nAverage Implementation Shortfall: ${:,.2f} \n'.format(avg_shortfall))

    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)

    # Include fee parameters in results if using HMF environment
    fee_args = {}
    if args.env == "HMF":
        fee_args = {
            'fixed_fee': args.fixed_fee,
            'proportional_fee': args.proportional_fee
        }

    # Save results to CSV
    save_results(
        agent_name='DDPG',
        env_name = args.env,
        reward_func=args.reward,
        avg_implementation_shortfall=avg_shortfall,
        **fee_args
    )
    
    # Save plots
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Modified plotting calls that handle saving
    def save_plot(plot_func, filename, *args, **kwargs):
        plt.figure()
        plot_func(*args, **kwargs)
        plt.savefig(f'results/{filename}_{timestamp}.png', bbox_inches='tight')
        plt.close()
    
    # Save training performance plot
    save_plot(
        plot_training_performance, 
        'training_performance', 
        shortfall_hist, 
        window_size=100, 
        figsize=(10, 6)
    )
    
    # Save training losses plot
    save_plot(
        plot_training_losses, 
        'training_losses', 
        agent, 
        window_size=100, 
        figsize=(12, 6)
    )

if __name__ == "__main__":
    main()