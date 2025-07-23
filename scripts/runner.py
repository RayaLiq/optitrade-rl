# runner.py
import argparse
import numpy as np
from collections import deque
import torch
import matplotlib.pyplot as plt

from envs.chriss_almgren import MarketEnvironment
from envs.gbm import GBMMarketEnvironment
from envs.hetson_merton import HMMarketEnvironment
from envs.hetson_metson_fees import HMFMarketEnvironment

from agents.ddpg_agent import Agent
from agents.sac_agent import SACAgent
from agents.td3_agent import TD3Agent

from rewards.core import REWARD_FN_MAP
from actions.scalers import transform_action, TRANSFORMS

import os
import sys
import pandas as pd
from datetime import datetime

from utils.utils import plot_training_losses
from utils.utils import plot_training_performance

def parse_args():
    
    parser = argparse.ArgumentParser(description='DDPG Agent for Optimal Trade Execution')

    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--agent', type=str, default="ddpg", choices=["ddpg", "td3", "sac"], help="RL agent type")
    parser.add_argument('--reward', nargs='+', default=['ac_utility'], choices=list(REWARD_FN_MAP.keys()), help='Reward function(s)')
    parser.add_argument('--env', type=str, default="AC", choices=["AC", "GBM", "HM", "HMF"], help='Market environment type')
    parser.add_argument('--action', nargs='+', default=['linear'], choices=list(TRANSFORMS.keys()), help="Agent action strategy/strategies")
    
    agent_sac_group = parser.add_argument_group('sac agent parameters', 'Parameters specific to sac agent')

    
    agent_td3_group = parser.add_argument_group('td3 agent parameters', 'Parameters specific to td3 agent')
    agent_td3_group.add_argument('--policy_delay', type=int, default=2)

    # Fee arguments (only used with HMF environment)
    fee_group = parser.add_argument_group('HMF environment fees', 'Parameters specific to Heston-Merton with Fees environment')
    fee_group.add_argument('--fixed_fee', type=float, default=10.0,help='Fixed fee per trade (only for HMF environment)')
    fee_group.add_argument('--proportional_fee', type=float, default=0.001,help='Proportional fee rate (only for HMF environment)')

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


def save_results(agent_name, env_name, reward_func, action_strategy, avg_implementation_shortfall,
                 fixed_fee=None, proportional_fee=None, filename='results/results.csv'):
    
    os.makedirs('results', exist_ok=True)

    data = {
        'agent': [agent_name],
        'env': [env_name],
        'reward_func': [reward_func],
        'action_strategy': [action_strategy],  # NEW: Add action strategy
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

    # NEW: Loop through all combinations
    for reward_function in args.reward:

        for action_strategy in args.action:

            print(f"\n{'='*85}")
            print(f"STARTING TRAINING: Agent={args.agent}, Environment={args.env}, Reward={reward_function}, Action={action_strategy}")
            print(f"{'='*85}")

            if args.env == "AC":
                # Initialize environment
                env = MarketEnvironment(
                    randomSeed=args.seed,
                    lqd_time=args.lqd_time,
                    num_tr=args.num_tr,
                    lambd=args.lambd,
                    reward_fn=reward_function
                )

            elif args.env == "GBM":  
                # Initialize environment
                env = GBMMarketEnvironment(
                    randomSeed=args.seed,
                    lqd_time=args.lqd_time,
                    num_tr=args.num_tr,
                    lambd=args.lambd,
                    reward_fn=reward_function
                )
                
            elif args.env == "HM":  
                # Initialize environment
                env = HMMarketEnvironment(
                    randomSeed=args.seed,
                    lqd_time=args.lqd_time,
                    num_tr=args.num_tr,
                    lambd=args.lambd,
                    reward_fn=reward_function
                )

            elif args.env == "HMF":  
                # Initialize environment
                env = HMFMarketEnvironment(
                    randomSeed=args.seed,
                    lqd_time=args.lqd_time,
                    num_tr=args.num_tr,
                    lambd=args.lambd,
                    reward_fn=reward_function,
                    fixed_fee=args.fixed_fee,
                    proportional_fee=args.proportional_fee
                )                               
            else:
                raise ValueError(f"Unknown environment type: {args.env}")


            if args.agent == "ddpg":
                # Initialize agent
                agent = Agent(
                    state_size=env.observation_space_dimension(),
                    action_size=env.action_space_dimension(),
                    random_seed=args.seed
                )

            elif args.agent == "td3":    
                # Initialize agent
                agent = TD3Agent(state_size=env.observation_space_dimension(), 
                                 action_size=env.action_space_dimension(), 
                                 random_seed=args.seed
                                 )                              
            elif args.agent == "sac":    
                # Initialize agent
                agent = SACAgent(state_size=env.observation_space_dimension(), 
                                 action_size=env.action_space_dimension(), 
                                 random_seed=args.seed
                                 )  
                
            else:
                raise ValueError(f"Unknown agent type: {args.agent}. Use 'ddpg', 'sac', or 'td3'.")                

            
            episodes = args.episodes
            lqt = args.lqd_time
            n_trades = args.num_tr
            tr = args.lambd

            shortfall_hist = np.array([])
            shortfall_deque = deque(maxlen=100)
            
            print(f"Starting training with {args.episodes} episodes...")

            print(f"Parameters: seed={args.seed}, reward={reward_function}, action={action_strategy}, lqd_time={args.lqd_time}, "
                f"num_tr={args.num_tr}, lambd={args.lambd}")
            
            if args.env == "HMF":
                print(f"Fee parameters: fixed_fee={args.fixed_fee}, proportional_fee={args.proportional_fee}")


            for episode in range(episodes):
                # Reset environment with new seed for each episode
                cur_state = env.reset(seed = episode, reward_fn=reward_function, liquid_time = lqt, num_trades = n_trades, lamb = tr)

                # set the environment to make transactions
                env.start_transactions()
                for i in range(n_trades + 1):

                    # Predict the best action for the current state. 
                    raw_action = agent.act(cur_state)
                    action = transform_action(raw_action, env, action_strategy)
                    
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
                agent_name=args.agent,
                env_name = args.env,
                reward_func=reward_function,
                action_strategy=action_strategy,
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
                f'training_perf_{args.agent}_{args.env}_{reward_function}_{action_strategy}.png', 
                shortfall_hist, 
                window_size=100, 
                figsize=(12, 6)
            )
            
            # Save training losses plot
            save_plot(
                plot_training_losses, 
                f'training_loss_{args.agent}_{args.env}_{reward_function}_{action_strategy}.png', 
                agent, 
                window_size=100, 
                figsize=(12, 6)
            )

if __name__ == "__main__":
    main()