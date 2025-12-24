import gymnasium as gym
import highway_env
from stable_baselines3 import DQN
from sb3_contrib import QRDQN
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

# --- CONFIGURATION ---
# Use N=100 for the final report to ensure statistical significance
N_EPISODES = 200  
RESULTS_DIR = "../../results/exp1_baseline/"
os.makedirs(RESULTS_DIR, exist_ok=True)

MODELS = {
    "DQN": "../../models/exp1_DQN_s1.zip",
    "QR-DQN": "../../models/exp1_QRDQN_s1.zip"
}

def get_test_env():
    # Identical physics to training, but NO rendering for speed
    config = {
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 15,
            "features": ["presence", "x", "y", "vx", "vy"],
            "features_range": {"x": [-100, 100], "y": [-100, 100], "vx": [-20, 20], "vy": [-20, 20]},
            "absolute": False
        },
        "simulation_frequency": 15,
        "policy_frequency": 1,
        "duration": 40,
        "lanes_count": 4, # Explicitly matching training
    }
    return gym.make("highway-v0", render_mode=None, config=config)

def test_model(algo, path):
    if not os.path.exists(path):
        print(f"⚠️  Model not found: {path}")
        return None

    print(f"Testing {algo} over {N_EPISODES} episodes...")
    env = get_test_env()
    
    # Load Model
    if algo == "DQN":
        model = DQN.load(path, env=env)
    elif algo == "QR-DQN":
        model = QRDQN.load(path, env=env)
    
    # Storage
    episode_rewards = []
    episode_speeds = []
    crashes = 0
    
    for i in range(N_EPISODES):
        obs, _ = env.reset()
        done = False
        truncated = False
        total_rew = 0
        speeds = []
        
        while not (done or truncated):
            # deterministic=True is CRITICAL for testing
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            
            total_rew += reward
            speeds.append(info['speed'])
            
            if done and info.get('crashed', False):
                crashes += 1
        
        episode_rewards.append(total_rew)
        episode_speeds.append(np.mean(speeds))
        
        # Progress bar
        if (i+1) % 10 == 0:
            print(f"  {i+1}/{N_EPISODES} completed...")

    env.close()
    
    return {
        "Model": algo,
        "Crash Rate": crashes / N_EPISODES,
        "Avg Reward": np.mean(episode_rewards),
        "Std Reward": np.std(episode_rewards), # Stability
        "Avg Speed": np.mean(episode_speeds),
        "Raw Rewards": episode_rewards # For plotting
    }

if __name__ == "__main__":
    all_results = []
    raw_data_for_plot = []

    for algo, path in MODELS.items():
        res = test_model(algo, path)
        if res:
            all_results.append(res)
            # Flatten for plotting
            for r in res["Raw Rewards"]:
                raw_data_for_plot.append({"Model": algo, "Reward": r})

    # --- 1. Save Numerical Metrics ---
    df_metrics = pd.DataFrame(all_results).drop(columns=["Raw Rewards"])
    csv_path = f"{RESULTS_DIR}/exp1_metrics.csv"
    df_metrics.to_csv(csv_path, index=False)
    print("\n--- FINAL METRICS ---")
    print(df_metrics)

    # --- 2. Plot Safety (Crash Rate) ---
    plt.figure(figsize=(7, 5))
    sns.barplot(data=df_metrics, x="Model", y="Crash Rate", palette="Reds")
    plt.title(f"Safety Baseline (N={N_EPISODES})")
    plt.ylabel("Crash Probability (Lower is Better)")
    plt.ylim(0, 1.0)
    plt.savefig(f"{RESULTS_DIR}/exp1_crash_rate.png")
    plt.close()

    # --- 3. Plot Stability (Violin Plot) ---
    # Violin plots show the distribution shape better than box plots
    df_raw = pd.DataFrame(raw_data_for_plot)
    plt.figure(figsize=(7, 5))
    sns.violinplot(data=df_raw, x="Model", y="Reward", palette="Blues", inner="quartile")
    plt.title(f"Reward Stability (N={N_EPISODES})")
    plt.ylabel("Total Episode Reward")
    plt.savefig(f"{RESULTS_DIR}/exp1_reward_distribution.png")
    plt.close()

    print(f"\n Testing Complete. Results saved to {RESULTS_DIR}")