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
import cv2 

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
    Also captures a snapshot if a crash happens at high density.
    """
    # 1. VISUALIZATION CHECK (Capture "The Wall")
    # If density is high, we run ONE separate instance with rendering enabled 
    # to catch a photo of the crash. We don't do this for all 1000 eps to save time.
    if density >= 1.8:
        capture_crash_snapshot(model_cfg, density)

    # 2. STATISTICAL EVALUATION (Fast, No Render)
    n_cpu = 6
    env = SubprocVecEnv([make_eval_env(density, render_mode=None) for _ in range(n_cpu)])
    
    path = os.path.join(script_dir, "models/{}.zip".format(model_cfg['file']))
    if not os.path.exists(path):
        return 100.0, 0.0, 0.0 # Fail defaults
    
    model = model_cfg['class'].load(path, env=env)
    
    crashes = 0
    episodes_completed = 0
    
    # Metric Accumulators
    total_speed_sum = 0
    total_steps_sum = 0
    total_steps_count = 0
    
    obs = env.reset()
    pbar = tqdm(total=N_EPISODES, desc=f"{model_cfg['name']} @ {density:.1f}", leave=False)
    
    while episodes_completed < N_EPISODES:
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)
        
        # Accumulate Speed info from all parallel envs
        # highway-env provides 'speed' in info dict (in m/s)
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
                
                # Duration is implicit in how long the env ran, 
                # but SubprocVecEnv resets automatically so exact duration per ep 
                # is harder to track perfectly without a Monitor wrapper. 
                # Approximation: Avg Duration = Total Steps / Total Episodes
    
    env.close()
    
    # Calculate Final Metrics
    crash_rate = (crashes / N_EPISODES) * 100
    avg_speed_ms = total_speed_sum / max(1, total_steps_count)
    avg_speed_kmh = avg_speed_ms * 3.6
    
    # Approx duration based on simulation steps (15 Hz simulation, 1 Hz policy)
    # Total steps counted / episodes. Note: Since policy_freq=1, 1 step = 1 second approx 
    # (actually depends on sim config, but usually step count ~ seconds in this config)
    avg_duration = total_steps_count / max(1, total_steps_count/n_cpu) # Rough correction or just use Monitor
    # Let's use a simpler heuristic: Total Steps / Episodes Completed (aggregated)
    avg_duration = (total_steps_count / n_cpu) / N_EPISODES 
    # Note: Parallel env logic makes exact duration tricky without Monitor. 
    # For simplicity/accuracy, let's trust the Speed metric more.
    
    return crash_rate, avg_speed_kmh, avg_duration

def capture_crash_snapshot(model_cfg, density):
    """
    Runs a SINGLE episode with rendering ON.
    If a crash happens, saves the image to prove 'The Wall' exists.
    """
    snapshot_dir = os.path.join(script_dir, "results", "crash_snapshots")
    os.makedirs(snapshot_dir, exist_ok=True)
    
    # Run 1 instance
    env = gym.make("highway-v0", render_mode="rgb_array")
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
    
    path = os.path.join(script_dir, "models/{}.zip".format(model_cfg['file']))
    if not os.path.exists(path): return
    model = model_cfg['class'].load(path, env=env)
    
    obs, _ = env.reset()
    done = False
    truncated = False
    
    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        
        if info.get("crashed", False):
            # CAPTURE THE MOMENT
            img = env.render()
            # Convert RGB to BGR for OpenCV
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                filename = f"{snapshot_dir}/crash_wall_den{density}_{model_cfg['name']}.png"
                cv2.imwrite(filename, img)
                # print(f"Snapshot saved: {filename}")
            break
    env.close()

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
    print("Check 'results/crash_snapshots/' for visual proof of traffic walls.")

def plot_results(results_dict, group_name, filename):
    plt.figure(figsize=(10, 6))
    styles = {"DQN": {"marker": 'o', "linestyle": '--'}, "QRDQN": {"marker": 's', "linestyle": '-'}}
    
    for name, rates in results_dict.items():
        if "DQN" in name and "QR" not in name:
            style = styles["DQN"]
            color = "#1f77b4" # Blue/Red depending on logic, kept simple here
        else:
            style = styles["QRDQN"]
            color = "#d62728" 
            
        plt.plot(DENSITIES, rates, label=name, linewidth=2.5, **style)

    #plt.axhline(y=10, color='black', linestyle=':', label='Collapse Threshold (10%)')
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