import gymnasium as gym
import highway_env
from stable_baselines3 import DQN
from sb3_contrib import QRDQN
from stable_baselines3.common.vec_env import SubprocVecEnv
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from collections import deque

# --- CONFIGURATION ---
N_EPISODES = 600
N_CPU = 24

# Delay in number of decisions (at 1 Hz, each decision is 1 second)
# 0 = no delay, 1 = 1 second delay, 2 = 2 seconds delay
DECISION_DELAYS = [0, 1, 2, 3, 4]

script_dir = os.path.abspath(".")
RESULTS_DIR = os.path.join(script_dir, "results/exp2_robustness/")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Define Models
MODELS = [
    {
        "name": "DQN (Aggressive)", 
        "class": DQN, 
        "file": os.path.join(script_dir, "models/exp1/E_Low_freq_aggr_DQN_s200.zip"),
        "baseline_crash_rate": 67.2
    },
    {
        "name": "QRDQN (Aggressive)", 
        "class": QRDQN, 
        "file": os.path.join(script_dir, "models/exp1/F_Low_freq_aggr_QRDQN_s200.zip"),
        "baseline_crash_rate": 13.8
    }
]

# --- DELAY WRAPPER ---
class ActionDelayWrapper(gym.Wrapper):
    """
    Wrapper that delays actions by a fixed number of decision steps.
    At 1 Hz control, delay=1 means 1 second latency.
    """
    def __init__(self, env, delay_steps=0):
        super().__init__(env)
        self.delay_steps = delay_steps
        # Initialize action buffer with IDLE action (action 1)
        self.action_buffer = deque([1] * (delay_steps + 1), maxlen=delay_steps + 1)
        
    def reset(self, **kwargs):
        # Reset action buffer to IDLE on episode reset
        self.action_buffer = deque([1] * (self.delay_steps + 1), 
                                   maxlen=self.delay_steps + 1)
        return self.env.reset(**kwargs)
    
    def step(self, action):
        """
        Store the new action and execute the delayed action.
        """
        # Add new action to buffer
        self.action_buffer.append(action)
        
        # Execute the oldest action (delayed by delay_steps)
        delayed_action = self.action_buffer[0]
        
        return self.env.step(delayed_action)

# --- ENVIRONMENT FACTORY ---
def make_delay_env(rank, delay_steps, seed=0):
    """
    Creates environment matching training configuration with action delay.
    """
    def _init():
        import gymnasium as gym
        import highway_env
        
        # CRITICAL: Match training configuration EXACTLY
        config = {
            "observation": {
                "type": "Kinematics",
                "vehicles_count": 15,
                "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
                "features_range": {
                    "x": [-100, 100],
                    "y": [-100, 100],
                    "vx": [-20, 20],
                    "vy": [-20, 20]
                },
                "absolute": False,
                "order": "sorted"  # ✓✓✓ CRITICAL
            },
            "action": {"type": "DiscreteMetaAction"},
            "lanes_count": 4,
            "vehicles_count": 50,
            "vehicles_density": 1.5,
            "duration": 60,
            
            # ✓✓✓ CRITICAL: Must match training
            "simulation_frequency": 15,
            "policy_frequency": 1,  # 1 Hz (NOT 15 Hz!)
            
            # Aggressive reward parameters
            "collision_reward": -15,
            "right_lane_reward": 0,
            "high_speed_reward": 0.5,
            "lane_change_reward": 0.1,
            "reward_speed_range": [25, 35],
            "normalize_reward": False
        }
        
        env = gym.make("highway-v0", render_mode=None)
        env.unwrapped.configure(config)
        
        # Wrap with action delay
        env = ActionDelayWrapper(env, delay_steps)
        
        env.reset(seed=seed + rank)
        return env
    return _init

# --- EVALUATION ---
def evaluate_delay_robustness():
    """
    Evaluate models under different action delays.
    """
    results = []
    
    print(f"\n{'='*70}")
    print(f"  EXPERIMENT 2: ROBUSTNESS TO SYSTEM LATENCY")
    print(f"{'='*70}")
    print(f"Episodes per delay: {N_EPISODES}")
    print(f"Parallel environments: {N_CPU}")
    print(f"Decision delays (at 1 Hz): {DECISION_DELAYS}")
    print(f"{'='*70}\n")
    
    for model_cfg in MODELS:
        model_name = model_cfg['name']
        model_file = model_cfg['file']
        baseline_crash = model_cfg['baseline_crash_rate']
        
        print(f"\n{'─'*70}")
        print(f"Testing: {model_name}")
        print(f"Baseline crash rate (no delay): {baseline_crash:.1f}%")
        print(f"{'─'*70}")
        
        for delay in DECISION_DELAYS:
            delay_seconds = delay * 1.0  # At 1 Hz, each delay step = 1 second
            print(f"\nDelay = {delay} steps ({delay_seconds:.1f}s)...")
            
            # Create environments with this delay
            env = SubprocVecEnv([
                make_delay_env(i, delay, seed=42) 
                for i in range(N_CPU)
            ])
            
            # Load model
            try:
                model = model_cfg['class'].load(model_file, env=env)
                print(f"  ✓ Model loaded")
            except Exception as e:
                print(f"  ✗ Error loading model: {e}")
                env.close()
                continue
            
            # Evaluation loop
            crashes = 0
            episodes_completed = 0
            total_rewards = []
            total_speeds = []
            
            current_rewards = np.zeros(N_CPU)
            current_speeds = [[] for _ in range(N_CPU)]
            current_crashes = np.zeros(N_CPU, dtype=bool)
            
            obs = env.reset()
            
            pbar = tqdm(
                total=N_EPISODES, 
                desc=f"  Delay={delay}s", 
                leave=False,
                ncols=80
            )
            
            while episodes_completed < N_EPISODES:
                # Get actions from model
                # The delay wrapper handles the buffering automatically
                actions, _ = model.predict(obs, deterministic=True)
                
                # Step environment (delay wrapper applies the buffered action)
                obs, rewards, dones, infos = env.step(actions)
                
                # Process each parallel environment
                for i in range(N_CPU):
                    current_rewards[i] += rewards[i]
                    
                    # Extract speed
                    speed = infos[i].get('speed', 0)
                    current_speeds[i].append(speed)
                    
                    # Check for crash
                    if infos[i].get('crashed', False):
                        current_crashes[i] = True
                    
                    # Episode finished
                    if dones[i]:
                        if episodes_completed < N_EPISODES:
                            # Record results
                            total_rewards.append(current_rewards[i])
                            avg_speed = np.mean(current_speeds[i]) if current_speeds[i] else 0
                            total_speeds.append(avg_speed)
                            
                            if current_crashes[i]:
                                crashes += 1
                            
                            episodes_completed += 1
                            pbar.update(1)
                        
                        # Reset tracking
                        current_rewards[i] = 0
                        current_speeds[i] = []
                        current_crashes[i] = False
            
            pbar.close()
            env.close()
            
            # Calculate metrics
            crash_rate = (crashes / N_EPISODES) * 100
            avg_reward = np.mean(total_rewards)
            std_reward = np.std(total_rewards)
            avg_speed = np.mean(total_speeds)
            
            # Store results
            results.append({
                "Model": model_name,
                "Delay_Steps": delay,
                "Delay_Seconds": delay_seconds,
                "Delay_ms": delay_seconds * 1000,  # For plotting
                "Crash_Rate": crash_rate,
                "Avg_Reward": avg_reward,
                "Std_Reward": std_reward,
                "Avg_Speed": avg_speed,
                "Degradation": crash_rate - baseline_crash
            })
            
            print(f"  Crash Rate: {crash_rate:.1f}% "
                  f"(Δ = +{crash_rate - baseline_crash:.1f}%)")
    
    # Save results
    df = pd.DataFrame(results)
    csv_path = os.path.join(RESULTS_DIR, "exp2_delay_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Results saved to: {csv_path}")
    
    # Generate plots
    generate_plots(df)
    
    # Print summary
    print_summary(df)
    
    return df

def generate_plots(df):
    """
    Generate visualization plots for delay robustness experiment.
    """
    sns.set_theme(style="whitegrid")
    
    # Plot 1: Crash Rate vs Delay
    plt.figure(figsize=(10, 6))
    for model_name in df['Model'].unique():
        model_data = df[df['Model'] == model_name]
        plt.plot(
            model_data['Delay_ms'],
            model_data['Crash_Rate'],
            marker='o', 
            linewidth=2.5,
            markersize=8,
            label=model_name
        )
    
    plt.xlabel('System Latency (ms)', fontsize=12)
    plt.ylabel('Crash Rate (%)', fontsize=12)
    plt.title('Robustness to Action Delay', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_path = os.path.join(RESULTS_DIR, "exp2_delay_crash_rate.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"✓ Plot saved to: {plot_path}")
    
    # Plot 2: Performance Degradation
    plt.figure(figsize=(10, 6))
    for model_name in df['Model'].unique():
        model_data = df[df['Model'] == model_name]
        plt.plot(
            model_data['Delay_ms'],
            model_data['Degradation'],
            marker='s', 
            linewidth=2.5,
            markersize=8,
            label=model_name
        )
    
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    plt.xlabel('System Latency (ms)', fontsize=12)
    plt.ylabel('Crash Rate Increase (%)', fontsize=12)
    plt.title('Performance Degradation vs Baseline', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_path = os.path.join(RESULTS_DIR, "exp2_delay_degradation.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"✓ Plot saved to: {plot_path}")

def print_summary(df):
    """
    Print summary statistics.
    """
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}\n")
    
    for model_name in df['Model'].unique():
        model_data = df[df['Model'] == model_name]
        baseline = model_data[model_data['Delay_Steps'] == 0]['Crash_Rate'].values[0]
        max_delay = model_data['Delay_Seconds'].max()
        max_crash = model_data[model_data['Delay_Steps'] == model_data['Delay_Steps'].max()]['Crash_Rate'].values[0]
        
        print(f"{model_name}:")
        print(f"  Baseline (0s delay):    {baseline:.1f}%")
        print(f"  Max delay ({max_delay:.1f}s):  {max_crash:.1f}%")
        print(f"  Total degradation:      +{max_crash - baseline:.1f}%")
        print()

if __name__ == "__main__":
    df = evaluate_delay_robustness()