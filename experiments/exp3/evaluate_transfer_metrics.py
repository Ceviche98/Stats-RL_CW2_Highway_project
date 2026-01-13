import gymnasium as gym
import highway_env
from sb3_contrib import QRDQN
from stable_baselines3 import DQN
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- CONFIG ---
script_dir = os.path.abspath(".")
N_EPISODES = 2000 # High sample size for tight Confidence Intervals

# Define your models here
models_to_test = [
    {"name": "DQN",   "type": DQN,   "path": "exp3_merge_DQN_s500"},
    {"name": "QRDQN", "type": QRDQN, "path": "exp3_merge_QRDQN_s500"}
]

def get_env_config(env_type):
    """Returns the observation config used in training."""
    base_config = {
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 15,
            "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
            "features_range": {
                "x": [-100, 100], "y": [-100, 100], "vx": [-20, 20], "vy": [-20, 20]
            },
            "absolute": False,
            "order": "sorted"
        },
        "action": { "type": "DiscreteMetaAction" },
        "simulation_frequency": 15,
        "policy_frequency": 1, 
    }
    
    if env_type == "merge":
        base_config["vehicles_density"] = 1.5
        base_config["duration"] = 20
    elif env_type == "roundabout":
        base_config["duration"] = 25 
        
    return base_config

def evaluate_model(model_info, env_name):
    """Evaluates a single model on a single environment."""
    env = gym.make(env_name)
    config = get_env_config("merge" if "merge" in env_name else "roundabout")
    env.unwrapped.configure(config)
    env.reset() # Critical fix for observation space

    # Load Model
    path = os.path.join(script_dir, "models/{}.zip".format(model_info["path"]))
    if not os.path.exists(path):
        print(f"Warning: {path} not found. Returning 0.")
        return 0, 0

    # Load with correct class (DQN or QRDQN)
    model = model_info["type"].load(path, env=env)

    success_count = 0
    
    print(f"Evaluating {model_info['name']} on {env_name}...")
    for _ in tqdm(range(N_EPISODES)):
        obs, info = env.reset()
        done = False
        truncated = False
        crashed = False
        
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            if info.get("crashed", False):
                crashed = True
                break
        
        if not crashed:
            success_count += 1
            
    env.close()
    
    success_rate = success_count / N_EPISODES
    # Calculate 95% Confidence Interval (Wald Interval)
    # Margin of Error = 1.96 * sqrt(p(1-p)/n)
    error = 1.96 * np.sqrt((success_rate * (1 - success_rate)) / N_EPISODES)
    
    return success_rate * 100, error * 100

def generate_comparison_graph():
    # Data storage
    results = {
        "Merge (Source)": [],
        "Roundabout (Target)": []
    }
    errors = {
        "Merge (Source)": [],
        "Roundabout (Target)": []
    }
    
    # Run Evaluations
    for env_name in ["merge-v0", "roundabout-v0"]:
        key = "Merge (Source)" if "merge" in env_name else "Roundabout (Target)"
        for model in models_to_test:
            rate, err = evaluate_model(model, env_name)
            results[key].append(rate)
            errors[key].append(err)

    # Plotting
    labels = ["Merge-v0 (Source)", "Roundabout-v0 (Target)"]
    dqn_means = [results["Merge (Source)"][0], results["Roundabout (Target)"][0]]
    dqn_errs = [errors["Merge (Source)"][0], errors["Roundabout (Target)"][0]]
    
    qrdqn_means = [results["Merge (Source)"][1], results["Roundabout (Target)"][1]]
    qrdqn_errs = [errors["Merge (Source)"][1], errors["Roundabout (Target)"][1]]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, dqn_means, width, yerr=dqn_errs, label='DQN', color='#1f77b4', capsize=5)
    rects2 = ax.bar(x + width/2, qrdqn_means, width, yerr=qrdqn_errs, label='QRDQN', color='#2ca02c', capsize=5)

    ax.set_ylabel('Success Rate (%)')
    ax.set_title('Zero-Shot Transfer: Gap Acceptance Generalization\n(N=1000 Episodes per Bar)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 105)
    ax.legend()

    # Label Bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), 
                        textcoords="offset points",
                        ha='center', va='bottom', fontweight='bold')

    autolabel(rects1)
    autolabel(rects2)

    save_path = os.path.join(script_dir, "results/transfer_comparison_CI.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"Graph saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    generate_comparison_graph()