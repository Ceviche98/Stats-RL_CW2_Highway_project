import gymnasium as gym
import highway_env
from sb3_contrib import QRDQN
import os
import cv2
import numpy as np
import imageio

# --- CONFIGURATION ---
script_dir = os.path.abspath(".")

# ENSURE THIS MATCHES YOUR SAVED MODEL FILENAME
model_name = "exp3_merge_QRDQN_s500" 
MODEL_PATH = os.path.join(script_dir, "models/{}.zip".format(model_name))
VIDEO_OUTPUT = os.path.join(script_dir, "results/videos/speedometer_merge_verification.mp4")

def record_merge_with_speedometer():
    # 1. Config matches Experiment 3 Aggressive
    env_config = {
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
        "vehicles_density": 1.5,
        "duration": 20, 
        "simulation_frequency": 15,
        "policy_frequency": 15, # Smooth video
        
        # --- AGGRESSIVE REWARDS ---
        "collision_reward": -15,    
        "high_speed_reward": 0.7,   
        "reward_speed_range": [23, 30], # [83, 108] km/h
        "normalize_reward": False
    }

    # 2. Create Environment
    env = gym.make("merge-v0", render_mode="rgb_array")
    env.unwrapped.configure(env_config)
    
    # --- CRITICAL FIX: RESET ENV TO APPLY CONFIGURATION ---
    # This updates the observation_space from (5,5) to (15,7)
    env.reset() 
    # -----------------------------------------------------

    # 3. Load Model
    print(f"Loading model from {MODEL_PATH}...")
    if not os.path.exists(MODEL_PATH):
        print("Error: Model not found.")
        return
    
    # Now this will work because env.observation_space is correct
    model = QRDQN.load(MODEL_PATH, env=env)

    frames = []
    
    # --- RECORD 3 EPISODES ---
    for episode in range(3):
        print(f"Recording Episode {episode + 1}/3...")
        obs, info = env.reset()
        done = False
        truncated = False
        
        while not (done or truncated):
            # Get Action
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            
            # Capture Frame
            frame = env.render()
            frame = np.ascontiguousarray(frame, dtype=np.uint8)
            
            # --- SPEEDOMETER LOGIC ---
            # 1. Get Speed
            try:
                speed_ms = env.unwrapped.vehicle.speed
            except:
                speed_ms = 0
            speed_kmh = speed_ms * 3.6
            
            # 2. Get Targets
            target_min = env_config["reward_speed_range"][0] * 3.6 # 82.8 km/h
            target_max = env_config["reward_speed_range"][1] * 3.6 # 108 km/h
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            
            # Line 1: Speed
            text_speed = f"Speed: {speed_kmh:.1f} km/h"
            
            # Color Logic
            if speed_kmh > target_max:
                color = (0, 0, 255) # Red (Reckless)
            elif speed_kmh >= target_min:
                color = (0, 255, 0) # Green (Good)
            else:
                color = (255, 0, 0) # Red (Too Slow / Danger on Ramp)
            
            # Draw Text with Outline
            cv2.putText(frame, text_speed, (20, 30), font, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, text_speed, (20, 30), font, 0.6, color, 2, cv2.LINE_AA)

            # Line 2: Episode Count
            text_ep = f"Ep: {episode+1}/3 | Target: {target_min:.0f}+"
            cv2.putText(frame, text_ep, (20, 55), font, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, text_ep, (20, 55), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            frames.append(frame)

    env.close()

    # 5. Save Video
    print(f"Saving video with {len(frames)} frames...")
    os.makedirs(os.path.dirname(VIDEO_OUTPUT), exist_ok=True)
    imageio.mimsave(VIDEO_OUTPUT, frames, fps=15)
    print(f"Video saved to: {VIDEO_OUTPUT}")

if __name__ == "__main__":
    record_merge_with_speedometer()