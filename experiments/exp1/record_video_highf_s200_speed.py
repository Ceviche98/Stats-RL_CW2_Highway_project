import gymnasium as gym
import highway_env
from sb3_contrib import QRDQN
from stable_baselines3 import DQN
import os
import imageio
import numpy as np

# --- CONFIGURATION ---
# Path to your trained model file (without .zip)
# EXAMPLE: "models/exp1c_aggressive_highfreq_QRDQN_s1000"
MODEL_PATH = "models/exp1c_aggressive_highfreq_QRDQN_s200.zip" 
MODEL_TYPE = "QRDQN"  # "DQN" or "QRDQN"
MODE = "aggressive"   # "aggressive" or "conservative"
VIDEO_NAME = "debug_highfreq_run.mp4"

# --- RE-CREATE THE EXACT ENVIRONMENT ---
# We must use the EXACT config used during training
def make_eval_env(mode="aggressive"):
    env_name = "highway-v0"
    config = {
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 15,
            "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
            "features_range": { "x": [-100, 100], "y": [-100, 100], "vx": [-20, 20], "vy": [-20, 20] },
            "absolute": False, 
            "order": "sorted"
        },
        "action": { "type": "DiscreteMetaAction" },
        "lanes_count": 4,
        "vehicles_count": 50,
        "vehicles_density": 1.5, 
        "duration": 60,              # Full 60 seconds
        "simulation_frequency": 15,  # Physics updates 15 times/sec
        "policy_frequency": 5,       # Agent decides 5 times/sec
        "normalize_reward": False,
        
        # VISUALIZATION SETTINGS
        "screen_width": 600,
        "screen_height": 300,
        "centering_position": [0.3, 0.5],
        "scaling": 5.5,
    }

    # Mode-specific config (same as training)
    if mode == "aggressive":
        config.update({
            "collision_reward": -30,
            "right_lane_reward": 0,    
            "high_speed_reward": 0.8,
            "lane_change_reward": 0,
            "reward_speed_range": [15, 35] 
        })
    else:
        config.update({
            "collision_reward": -40,
            "right_lane_reward": 0,
            "high_speed_reward": 0.3,
            "lane_change_reward": -0.1,
            "reward_speed_range": [10, 30]
        })

    env = gym.make(env_name, render_mode="rgb_array")
    env.unwrapped.configure(config)
    env.reset()
    return env

def record_video():
    print(f"--- Loading Model: {MODEL_PATH} ---")
    
    # 1. Create Env
    env = make_eval_env(MODE)
    
    # 2. Load Model
    if MODEL_TYPE == "QRDQN":
        model = QRDQN.load(MODEL_PATH, env=env)
    else:
        model = DQN.load(MODEL_PATH, env=env)

    # 3. Run Episode
    obs, _ = env.reset()
    done = False
    truncated = False
    frames = []
    
    step_counter = 0
    total_reward = 0
    
    print("--- Starting Simulation (Max 60s) ---")
    
    while not (done or truncated):
        # Predict action
        action, _ = model.predict(obs, deterministic=True)
        
        # Step environment
        # NOTE: With policy_freq=5 and sim_freq=15, 
        # this single step advances the physics by 3 frames (0.2 seconds).
        # The environment automatically holds the action ("idle" logic) for those frames.
        obs, reward, done, truncated, info = env.step(action)
        
        total_reward += reward
        step_counter += 1
        
        # Render frame
        frame = env.render()
        
        # Add text to frame (Step count + Speed)
        # (Optional: requires OpenCV, skipping for simplicity, just saving raw frame)
        frames.append(frame)

        # Print progress every second (5 steps)
        if step_counter % 5 == 0:
            speed = info['speed'] * 3.6 if 'speed' in info else 0
            print(f"Time: {step_counter/5:.1f}s | Speed: {speed:.1f} km/h | Reward: {reward:.2f}")

    env.close()
    
    # 4. Save Video
    print(f"--- Saving Video to {VIDEO_NAME} ---")
    # We save at 5 FPS because that's the policy frequency (what the agent sees)
    # If you want smooth physics playback, 15 FPS would be 'real time' but 
    # we only captured the policy steps.
    # To capture all physics frames, we'd need to modify the env, 
    # but 5 FPS is enough to diagnose decisions.
    imageio.mimsave(VIDEO_NAME, frames, fps=5)
    print("Done!")

if __name__ == "__main__":
    record_video()