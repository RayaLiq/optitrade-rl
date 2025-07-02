import logging
import numpy as np
import pandas as pd
from tqdm import trange


import matplotlib.pyplot as plt
import seaborn as sns

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from syntheticChrissAlmgren_extended import MarketEnvironment
from ddpg_agent import Agent

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("DRL-Reward-Test")

# Define reward functions including the new ones from research
reward_functions = [
    "ac_utility",
    "capture",
    "custom_penalty",
    "final_shortfall",
    "stepwise_shortfall",
    "hybrid_shortfall_risk",
    "smoothness_penalty",
    "baseline_relative",
    "inv_time_penalty",
    "risk_adjusted_utility"
]

def run_experiment(reward_function: str, n_episodes: int = 1000, seed: int = 0):
    logger.info(f"Running experiment with reward_function = '{reward_function}'")

    env = MarketEnvironment(randomSeed=seed)
    state_size = env.observation_space_dimension()
    action_size = env.action_space_dimension()

    agent = Agent(state_size=state_size, action_size=action_size, random_seed=seed)

    episode_rewards = []
    episode_shortfalls = []

    for ep in trange(n_episodes, desc=f"Reward={reward_function}"):
        state = env.reset(seed + ep)
        env.start_transactions()
        agent.reset()

        total_reward = 0
        done = False
        while not done:
            action = agent.act(state, add_noise=True)
            next_state, reward, done, info = env.step(action, reward_function=reward_function)

            # Adjust reward for special cases
            if reward_function == "final_shortfall" and done:
                reward = -info.implementation_shortfall / (env.total_shares * env.startingPrice)
            elif reward_function == "baseline_relative" and done:
                ac_shortfall = env.get_AC_expected_shortfall(env.total_shares)
                reward = (ac_shortfall - info.implementation_shortfall) / ac_shortfall
            elif reward_function == "stepwise_shortfall":
                reward = reward  # already dense shortfall-based
            elif reward_function == "hybrid_shortfall_risk":
                impact_penalty = 0.00001 * (info.share_to_sell_now ** 2) if hasattr(info, 'share_to_sell_now') else 0
                reward = ((env.startingPrice - info.exec_price) * info.share_to_sell_now - impact_penalty) / (env.total_shares * env.startingPrice)
            elif reward_function == "smoothness_penalty":
                smooth_penalty = abs(action - agent.last_action) if hasattr(agent, 'last_action') else 0
                reward = float(reward) - 0.01 * smooth_penalty
                agent.last_action = action

            reward = float(reward)
            agent.step(state, action, reward, next_state, done)
            total_reward += reward
            state = next_state

        shortfall = info.implementation_shortfall
        episode_rewards.append(total_reward)
        episode_shortfalls.append(shortfall)

        if ep % 1000 == 0:
            logger.info(f"Episode {ep + 1}: Reward = {total_reward:.4f}, Shortfall = ${shortfall:,.2f}")

    avg_reward = np.mean(episode_rewards)
    avg_shortfall = np.mean(episode_shortfalls)
    logger.info(f"[{reward_function}] Avg Reward: {avg_reward:.4f}, Avg Shortfall: ${avg_shortfall:,.2f}")

    # Save to CSV
    df = pd.DataFrame({
        "episode": list(range(n_episodes)),
        "reward": episode_rewards,
        "shortfall": episode_shortfalls
    })
    df.to_csv(f"reward_{reward_function}.csv", index=False)

    return episode_rewards, episode_shortfalls

# Run and save results
results = {}
for rf in reward_functions:
    rewards, shortfalls = run_experiment(rf, n_episodes=1000)
    results[rf] = {"rewards": rewards, "shortfalls": shortfalls}

# ====== Convert results dictionary to DataFrame ======
all_data = []
for reward_type, data in results.items():
    for r in data["rewards"]:
        all_data.append({"Reward Function": reward_type, "Episode Reward": r})
    for s in data["shortfalls"]:
        all_data.append({"Reward Function": reward_type, "Implementation Shortfall": s})

# Use DataFrame split for analysis
df_rewards = pd.DataFrame([
    {"Reward Function": rf, "Episode Reward": r}
    for rf, res in results.items()
    for r in res["rewards"]
])
df_shortfalls = pd.DataFrame([
    {"Reward Function": rf, "Implementation Shortfall": s}
    for rf, res in results.items()
    for s in res["shortfalls"]
])

# ====== Save to CSV ======
df_rewards.to_csv("episode_rewards.csv", index=False)
df_shortfalls.to_csv("implementation_shortfalls.csv", index=False)

'''
# ====== Plot Average Reward with Std ======
plt.figure(figsize=(15, 10))
sns.barplot(data=df_rewards, x="Reward Function", y="Episode Reward", errorbar='sd', palette="muted", legend=False, hue='Reward Function')
plt.title("Average Episode Reward by Reward Function")
plt.tight_layout()
plt.savefig("avg_rewards.png")
plt.show()

# ====== Plot Shortfall Distribution ======
plt.figure(figsize=(15, 10))
sns.boxplot(data=df_shortfalls, x="Reward Function", y="Implementation Shortfall", palette="coolwarm", legend=False, hue = 'Reward Function')
plt.title("Implementation Shortfall Distribution")
plt.tight_layout()
plt.savefig("shortfall_boxplot.png")
plt.show()
'''

# ====== Print Summary Statistics ======
summary = {}
for reward_type, data in results.items():
    reward_arr = np.array(data["rewards"])
    shortfall_arr = np.array(data["shortfalls"])
    summary[reward_type] = {
        "reward_mean": reward_arr.mean(),
        "reward_std": reward_arr.std(),
        "shortfall_mean": shortfall_arr.mean(),
        "shortfall_std": shortfall_arr.std()
    }

summary_df = pd.DataFrame(summary).T
summary_df.index.name = "Reward Function"
# Optional: Save summary
summary_df.to_csv("summary_statistics.csv")