from multiprocess import Pool
import subprocess
from pathlib import Path
import itertools
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- CONFIGURATION ---
agents = ["td3", "sac", "ddpg"]
rewards = ["permanent_impact_penalty", "sparse_reward"]
actions = ["linear", "square"]
envs = ["gbm", "heston_merton", "heston_merton_fees"]
episodes = 1000

csv_dir = Path("final_test_with_all_envs")
log_dir = Path("logs")

csv_dir.mkdir(exist_ok=True)
log_dir.mkdir(exist_ok=True)

def run_experiment(args):
    agent, reward, action, env = args
    tag = f"{agent}_{reward}_{action}_{env}"
    log_file = log_dir / f"{tag}.log"

    cmd = [
        sys.executable, "runner.py",
        "--agent", agent,
        "--env", env,
        "--reward", reward,
        "--action", action,
        "--episodes", str(episodes),
        "--csv-dir", str(csv_dir),
    ]

    with open(log_file, "w") as f:
        f.write(f"Running: {tag}\nCommand: {' '.join(cmd)}\n\n")
        print(f"🚀 [START] {tag}")
        result = subprocess.run(cmd, stdout=f, stderr=f, text=True)
        if result.returncode == 0:
            print(f"✅ [SUCCESS] {tag}")
        else:
            print(f"❌ [FAILED ] {tag} — check log: {log_file}")

def aggregate_results(result_dir):
    all_data = []
    for csv_file in Path(result_dir).glob("*.csv"):
        if "summary" in csv_file.name:
            continue
        try:
            df = pd.read_csv(csv_file)
            name_parts = csv_file.stem.split("_")
            if len(name_parts) < 4:
                continue  # skip malformed
            agent, reward, action = name_parts[0], name_parts[1], name_parts[2]
            env = "_".join(name_parts[3:])
            reward_mean = df["reward"].mean()
            reward_std = df["reward"].std()
            shortfall_mean = df["shortfall"].mean()
            shortfall_std = df["shortfall"].std()
            all_data.append({
                "Agent": agent,
                "Reward": reward,
                "Action": action,
                "Environment": env,
                "reward_mean": reward_mean,
                "reward_std": reward_std,
                "shortfall_mean": shortfall_mean,
                "shortfall_std": shortfall_std
            })
        except Exception as e:
            print(f"⚠️ Skipping {csv_file.name}: {e}")
    df_all = pd.DataFrame(all_data)
    df_all.to_csv(Path(result_dir) / "aggregate_summary.csv", index=False)
    print(f"✅ Summary saved to {result_dir}/aggregate_summary.csv")

if __name__ == "__main__":
    all_args = list(itertools.product(agents, rewards, actions, envs))
    with Pool(processes=36) as pool:
        pool.map(run_experiment, all_args)
    aggregate_results(csv_dir)
