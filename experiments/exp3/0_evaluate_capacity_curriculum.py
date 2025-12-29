import gymnasium as gym
import highway_env
from sb3_contrib import QRDQN
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import SubprocVecEnv
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import csv

# --- CONFIGURATION ---
script_dir = os.path.abspath(".")
DENSITIES = np.arange(1.5, 2.51, 0.1) 
N_EPISODES = 1000 

# DEFINE YOUR MODELS
MODELS_CONSERVATIVE = [
    {"name": "DQN (Conservative)",   "class": DQN,   "file": "exp1_conservative_DQN_s200", "color": "#1f77b4"}, 
    {"name": "QRDQN (Conservative)", "class": QRDQN, "file": "exp1_conservative_QRDQN_s200", "color": "#2ca02c"} 
]

MODELS_AGGRESSIVE = [
    {"name": "DQN (Aggressive)",     "class": DQN,   "file": "exp1_aggressive_DQN_s200", "color": "#d62728"}, 
    {"name": "QRDQN (Aggressive)",   "class": QRDQN, "file": "exp1_aggressive_QRDQN_s200", "color": "#9467bd"} 
]

# --- ENVIRONMENT FACTORY ---
def make_eval_env(density, render_mode=None):
    def _init():
        env = gym.make("highway-v0", render_mode=render_mode)
        config = {
            "observation": {
                "type": "Kinematics",
                "vehicles_count": 15,
                "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
                "features_range": { "x": [-100, 100], "y": [-100, 100], "vx": [-20, 20], "vy": [-20, 20] },
                "absolute": False, "order": "sorted"
            },
            "action": { "type": "DiscreteMetaAction" },
            "simulation_frequency": 15, "policy_frequency": 1,
            "duration": 60,
            "vehicles_density": density,
            "collision_reward": -15
        }
        env.unwrapped.configure(config)
        env.reset()
        return env
    return _init

def evaluate_metrics(model_cfg, density):
    """
    Calculates Crash Rate, Avg Speed, and Avg Duration.
    """
    
    # STATISTICAL EVALUATION (Fast, No Render)
    n_cpu = 6
    env = SubprocVecEnv([make_eval_env(density, render_mode=None) for _ in range(n_cpu)])
    
    path = os.path.join(script_dir, "models/{}.zip".format(model_cfg['file']))
    if not os.path.exists(path):
        print(f"Warning: Model not found at {path}")
        return 100.0, 0.0, 0.0 # Fail defaults
    
    model = model_cfg['class'].load(path, env=env)
    
    crashes = 0
    episodes_completed = 0
    
    # Metric Accumulators
    total_speed_sum = 0
    total_steps_count = 0
    
    obs = env.reset()
    pbar = tqdm(total=N_EPISODES, desc=f"{model_cfg['name']} @ {density:.1f}", leave=False)
    
    while episodes_completed < N_EPISODES:
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)
        
        # Accumulate Speed info from all parallel envs
        for info in infos:
            if "speed" in info:
                total_speed_sum += info['speed']
                total_steps_count += 1
        
        for i, done in enumerate(dones):
            if done:
                episodes_completed += 1
                pbar.update(1)
                
                # Check Crash
                if infos[i].get("crashed", False):
                    crashes += 1
                
                # Exit early if we hit the target count
                if episodes_completed >= N_EPISODES:
                    break
    
    env.close()
    
    # Calculate Final Metrics
    crash_rate = (crashes / N_EPISODES) * 100
    avg_speed_ms = total_speed_sum / max(1, total_steps_count)
    avg_speed_kmh = avg_speed_ms * 3.6
    
    # Approx duration
    avg_duration = (total_steps_count / n_cpu) / N_EPISODES 
    
    return crash_rate, avg_speed_kmh, avg_duration

def run_stress_test():
    # PREPARE CSV
    csv_path = os.path.join(script_dir, "results", "capacity_curriculum_full_metrics.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    csv_file = open(csv_path, mode='w', newline='')
    writer = csv.writer(csv_file)
    writer.writerow(["Experiment_Group", "Model_Type", "Density", "Crash_Rate", "Avg_Speed_kmh", "Avg_Duration_sec"])

    # 1. Evaluate CONSERVATIVE
    print("\n--- STRESS TEST: CONSERVATIVE MODELS ---")
    results_cons = {m['name']: [] for m in MODELS_CONSERVATIVE}
    
    for density in DENSITIES:
        for m in MODELS_CONSERVATIVE:
            rate, speed, dur = evaluate_metrics(m, density)
            results_cons[m['name']].append(rate)
            writer.writerow(["Conservative", m['name'], round(density, 1), round(rate, 2), round(speed, 2), round(dur, 2)])
            
    plot_results(results_cons, "Conservative", "results/stress_test_conservative.png")

    # 2. Evaluate AGGRESSIVE
    print("\n--- STRESS TEST: AGGRESSIVE MODELS ---")
    results_aggr = {m['name']: [] for m in MODELS_AGGRESSIVE}
    
    for density in DENSITIES:
        for m in MODELS_AGGRESSIVE:
            rate, speed, dur = evaluate_metrics(m, density)
            results_aggr[m['name']].append(rate)
            writer.writerow(["Aggressive", m['name'], round(density, 1), round(rate, 2), round(speed, 2), round(dur, 2)])
            
    plot_results(results_aggr, "Aggressive", "results/stress_test_aggressive.png")
    
    csv_file.close()
    print(f"\nSUCCESS: Full metrics saved to {csv_path}")

def plot_results(results_dict, group_name, filename):
    plt.figure(figsize=(10, 6))
    styles = {"DQN": {"marker": 'o', "linestyle": '--'}, "QRDQN": {"marker": 's', "linestyle": '-'}}
    
    for name, rates in results_dict.items():
        if "DQN" in name and "QR" not in name:
            style = styles["DQN"]
        else:
            style = styles["QRDQN"]
            
        plt.plot(DENSITIES, rates, label=name, linewidth=2.5, **style)

    plt.xlabel("Traffic Density")
    plt.ylabel("Crash Rate (%)")
    plt.title(f"Capacity Curriculum: {group_name} Models")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(-2, 102)
    plt.xticks(DENSITIES)
    
    plt.savefig(filename)
    plt.show()

if __name__ == "__main__":
    run_stress_test()