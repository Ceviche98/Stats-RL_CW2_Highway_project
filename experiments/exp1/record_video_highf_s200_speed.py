import gymnasium as gym
import highway_env
from sb3_contrib import QRDQN
from stable_baselines3 import DQN
import os
import imageio
import numpy as np
import cv2

# --- CONFIGURATION ---
script_dir = os.path.abspath(".")
# Updated to use best high-frequency models (I or J)
model_name = "I_High_freq_aggressive_QRDQN_s500_opt"
MODEL_PATH = os.path.join(script_dir, "models/exp1/{}.zip".format(model_name))
VIDEO_OUTPUT = os.path.join(script_dir, "results/videos/speedometer_highfreq_{}.mp4".format(model_name))

MODEL_TYPE = "QRDQN"  # "DQN" or "QRDQN"
MODE = "aggressive"   # "aggressive" or "conservative"

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
    
    print("--- Starting Simulation (Max 60s at 5 Hz) ---")
    
    while not (done or truncated):
        # Predict action
        action, _ = model.predict(obs, deterministic=True)
        
        # Step environment
        obs, reward, done, truncated, info = env.step(action)
        
        total_reward += reward
        step_counter += 1
        
        # Render frame
        frame = env.render()
        
        # --- ADD SPEEDOMETER OVERLAY ---
        frame = np.ascontiguousarray(frame, dtype=np.uint8)
        
        # Get speed info
        speed_ms = env.unwrapped.vehicle.speed
        speed_kmh = speed_ms * 3.6
        
        # Reward zone from config (high-freq: [15, 35] m/s)
        config_mode = make_eval_env(MODE)
        target_min_kmh = 15 * 3.6  # 54 km/h
        target_max_kmh = 35 * 3.6  # 126 km/h
        
        # Color logic
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_speed = f"Speed: {speed_kmh:.1f} km/h"
        
        if speed_kmh > target_max_kmh:
            color = (0, 0, 255)  # Red (Too fast)
        elif speed_kmh >= target_min_kmh:
            color = (0, 255, 0)  # Green (In reward zone)
        else:
            color = (255, 0, 0)  # Blue (Too slow)
        
        # Draw speedometer
        cv2.putText(frame, text_speed, (20, 30), font, 0.6, (0, 0, 0), 3, cv2.LINE_AA)  # Outline
        cv2.putText(frame, text_speed, (20, 30), font, 0.6, color, 2, cv2.LINE_AA)       # Text
        
        # Draw legend
        text_legend = f"Target: {target_min_kmh:.0f}-{target_max_kmh:.0f} km/h"
        cv2.putText(frame, text_legend, (20, 55), font, 0.5, (0, 0, 0), 3, cv2.LINE_AA)  # Outline
        cv2.putText(frame, text_legend, (20, 55), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)  # White
        
        frames.append(frame)

        # Print progress every second (5 steps)
        if step_counter % 5 == 0:
            print(f"Time: {step_counter/5:.1f}s | Speed: {speed_kmh:.1f} km/h | Reward: {reward:.2f}")

    env.close()
    
    # 4. Save Video
    print(f"--- Saving Video to {VIDEO_OUTPUT} ---")
    os.makedirs(os.path.dirname(VIDEO_OUTPUT), exist_ok=True)
    # Save at 5 FPS (matches policy frequency)
    imageio.mimsave(VIDEO_OUTPUT, frames, fps=5)
    print(f"✓ Done! Video saved to {VIDEO_OUTPUT}")
    print(f"Total reward: {total_reward:.2f}")

if __name__ == "__main__":
    record_video()