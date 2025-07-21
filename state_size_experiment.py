import numpy as np
from syntheticChrissAlmgren import MarketEnvironment
from runner import train_once
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import logging
import traceback
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('state_size_experiment.log'),
        logging.StreamHandler()
    ]
)

def setup_directories(base_path):
    """Ensure output directories exist"""
    try:
        os.makedirs(base_path, exist_ok=True)
        logging.info(f"Created output directory: {base_path}")
        return True
    except Exception as e:
        logging.error(f"Failed to create directory {base_path}: {str(e)}")
        return False

def run_state_size_experiment(state_sizes, episodes=1000, seed=42):
    results = []
    base_path = "state_size_results"
    
    if not setup_directories(base_path):
        return None
    
    for size in state_sizes:
        try:
            size_path = os.path.join(base_path, f"size_{size}")
            os.makedirs(size_path, exist_ok=True)
            
            logging.info(f"\n{'='*50}")
            logging.info(f"Starting experiment with state size {size}")
            logging.info(f"{'='*50}")
            
            # Create environment
            logging.info("Creating environment...")
            env = MarketEnvironment(randomSeed=seed, state_size=size)
            logging.info(f"Environment created with state dimension: {env.observation_space_dimension()}")
            
            # Train and evaluate
            logging.info("Starting training...")
            results_dict = train_once(
                env_name="ac_default",
                agent_name="ddpg",
                reward_fn="ac_utility",
                act_method="linear",
                episodes=episodes,
                seed=seed,
                csv_dir=Path(size_path)
            )
            
            if results_dict is None:
                logging.warning(f"Training returned None for size {size}")
                continue
                
            mean_reward, std_reward, mean_shortfall, std_shortfall, shortfall_history, _ = results_dict
            
            logging.info(f"Completed size {size} with shortfall: {mean_shortfall:.2f} ± {std_shortfall:.2f}")
            
            results.append({
                "state_size": size,
                "mean_reward": mean_reward,
                "std_reward": std_reward,
                "mean_shortfall": mean_shortfall,
                "std_shortfall": std_shortfall
            })
            
            # Save shortfall history
            history_path = os.path.join(size_path, "shortfall_history.csv")
            pd.DataFrame(shortfall_history, columns=["shortfall"]).to_csv(history_path, index=False)
            logging.info(f"Saved shortfall history to {history_path}")
            
        except Exception as e:
            logging.error(f"Error in state size {size}: {str(e)}")
            logging.error(traceback.format_exc())
            continue
    
    return pd.DataFrame(results) if results else None

def plot_results(results_df):
    try:
        plt.figure(figsize=(12, 6))
        
        plt.errorbar(
            results_df["state_size"],
            results_df["mean_shortfall"],
            yerr=results_df["std_shortfall"],
            fmt='-o',
            capsize=5,
            label="Implementation Shortfall"
        )
        
        plt.xlabel("State Size (logReturns window)")
        plt.ylabel("Mean Implementation Shortfall ($)")
        plt.title("Trading Performance by State Size")
        plt.xticks(results_df["state_size"])
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        
        plot_path = os.path.join("state_size_results", "state_size_comparison.png")
        plt.savefig(plot_path)
        plt.close()
        logging.info(f"Saved plot to {plot_path}")
        
    except Exception as e:
        logging.error(f"Error plotting results: {str(e)}")
        logging.error(traceback.format_exc())

if __name__ == "__main__":
    logging.info("Starting state size experiment")
    
    state_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 12]
    results = run_state_size_experiment(state_sizes)
    
    if results is not None and not results.empty:
        summary_path = os.path.join("state_size_results", "summary.csv")
        results.to_csv(summary_path, index=False)
        logging.info(f"Saved summary to {summary_path}")
        plot_results(results)
    else:
        logging.error("Experiment failed to produce results")
    
    logging.info("Experiment completed")