import gymnasium as gym
import highway_env
from sb3_contrib import QRDQN
import os
import cv2
import numpy as np
import imageio

# --- CONFIGURATION ---
script_dir = os.path.abspath(".")
model_name = "exp1_conservative_QRDQN_s200" 
MODEL_PATH = os.path.join(script_dir, "models/{}.zip".format(model_name))
VIDEO_OUTPUT = os.path.join(script_dir, "results/videos/speedometer_{}.mp4".format(model_name))

def record_with_speedometer():
    # 1. Config for Smooth Video (Policy Freq = 15)
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
            "lanes_count": 4,
            "vehicles_count": 50,
            "vehicles_density": 1.5, # Expert difficulty
            "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
            "duration": 60, 
            "simulation_frequency": 15,
            "policy_frequency": 15,
            
            # --- OPTIMIZED REWARDS (User: Gamma 0.9 Specs) ---
            "collision_reward": -20,    # The "Death Sentence"
            "right_lane_reward": 0,     # Encourages overtaking
            "high_speed_reward": 0.4,   # Overshadowed by crash penalty
            "lane_change_reward": 0, 
            "reward_speed_range": [20, 30],
            "normalize_reward": False
        }

    # 2. Create Environment
    env = gym.make("highway-v0", render_mode="rgb_array")
    env.unwrapped.configure(env_config)
    env.reset()

    # 3. Load Model
    print(f"Loading model from {MODEL_PATH}...")
    if not os.path.exists(MODEL_PATH):
        print("Error: Model not found.")
        return
    model = QRDQN.load(MODEL_PATH, env=env)

    # 4. Recording Loop
    frames = []
    obs, info = env.reset()
    done = False
    truncated = False
    
    print("Recording frame by frame...")
    
    while not (done or truncated):
        # Get Action
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        
        # Capture Frame
        frame = env.render()
        
        # --- FIX: MAKE MEMORY CONTIGUOUS FOR OPENCV ---
        # This fixes the "Layout of output array incompatible" error
        frame = np.ascontiguousarray(frame, dtype=np.uint8)
        
        # --- DATA CALCULATION ---
        # 1. Current Speed
        speed_ms = env.unwrapped.vehicle.speed
        speed_kmh = speed_ms * 3.6
        
        # 2. Reward Zone (From Config)
        # Config is [20, 30] m/s -> [72, 108] km/h
        target_min = env_config["reward_speed_range"][0] * 3.6
        target_max = env_config["reward_speed_range"][1] * 3.6
        
        # --- VISUALIZATION LOGIC ---
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Line 1: Current Speed
        text_speed = f"Speed: {speed_kmh:.1f} km/h"
        
        # Color Logic
        if speed_kmh > target_max:
            color = (0, 0, 255) # Red (Too fast, dangerous?)
        elif speed_kmh >= target_min:
            color = (0, 255, 0) # Green (In Reward Zone)
        else:
            color = (255, 0, 0) # Red/Blue (Too slow) (OpenCV uses BGR, but highway-env is RGB)
            # Actually highway-env returns RGB.
            # So (255, 0, 0) is RED. (0, 255, 0) is GREEN.
        
        # Draw Line 1 (Speed)
        cv2.putText(frame, text_speed, (20, 30), font, 0.6, (0, 0, 0), 3, cv2.LINE_AA) # Outline
        cv2.putText(frame, text_speed, (20, 30), font, 0.6, color, 2, cv2.LINE_AA)     # Text

        # Line 2: Reward Legend
        text_legend = f"Target: {target_min:.0f}-{target_max:.0f} km/h"
        cv2.putText(frame, text_legend, (20, 55), font, 0.5, (0, 0, 0), 3, cv2.LINE_AA) # Outline
        cv2.putText(frame, text_legend, (20, 55), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA) # White Text

        frames.append(frame)

    env.close()

    # 5. Save Video
    print(f"Saving video with {len(frames)} frames...")
    os.makedirs(os.path.dirname(VIDEO_OUTPUT), exist_ok=True)
    
    # Save at 15 FPS
    imageio.mimsave(VIDEO_OUTPUT, frames, fps=15)
    print(f"Video saved to: {VIDEO_OUTPUT}")

if __name__ == "__main__":
    record_with_speedometer()