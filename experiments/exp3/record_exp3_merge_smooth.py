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
VIDEO_OUTPUT = os.path.join(script_dir, "results/videos/speedometer_merge_smooth.mp4")

def record_merge_smooth():
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
        
        # TRICK: We set this to 15 so env.step() only moves 1/15th of a second.
        # We will handle the "1 Hz logic" inside the loop manually.
        "policy_frequency": 15, 
        
        # --- AGGRESSIVE REWARDS ---
        "collision_reward": -15,    
        "high_speed_reward": 0.7,   
        "reward_speed_range": [23, 30], # [83, 108] km/h
        "normalize_reward": False
    }

    # 2. Create Environment
    env = gym.make("merge-v0", render_mode="rgb_array")
    env.unwrapped.configure(env_config)
    
    # CRITICAL: Reset to apply config BEFORE loading model
    env.reset()

    # 3. Load Model
    print(f"Loading model from {MODEL_PATH}...")
    if not os.path.exists(MODEL_PATH):
        print("Error: Model not found.")
        return
    model = QRDQN.load(MODEL_PATH, env=env)

    frames = []
    
    # --- RECORD 3 EPISODES ---
    for episode in range(3):
        print(f"Recording Episode {episode + 1}/3...")
        obs, info = env.reset()
        done = False
        truncated = False
        
        # LOGIC VARS FOR SMOOTHING
        step_counter = 0
        current_action = 1 # Default IDLE
        
        while not (done or truncated):
            
            # --- THE JITTER FIX ---
            # Only ask the model for a new decision every 15 steps (1 second)
            # This matches the training frequency (1 Hz)
            if step_counter % 15 == 0:
                current_action, _ = model.predict(obs, deterministic=True)
            
            # Step the environment with the HELD action
            obs, reward, done, truncated, info = env.step(current_action)
            step_counter += 1
            
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
            target_min = env_config["reward_speed_range"][0] * 3.6 
            target_max = env_config["reward_speed_range"][1] * 3.6 
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            
            # Line 1: Speed
            text_speed = f"Speed: {speed_kmh:.1f} km/h"
            
            # Color Logic
            if speed_kmh > target_max:
                color = (0, 0, 255) 
            elif speed_kmh >= target_min:
                color = (0, 255, 0) # Green (Good)
            else:
                color = (255, 0, 0) 
            
            # Draw Text with Outline
            cv2.putText(frame, text_speed, (20, 30), font, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, text_speed, (20, 30), font, 0.6, color, 2, cv2.LINE_AA)

            # Line 2: Action Status
            # Show what the agent is currently "Holding"
            action_map = {0: "LANE LEFT", 1: "IDLE", 2: "LANE RIGHT", 3: "FASTER", 4: "SLOWER"}
            action_text = f"Act: {action_map.get(int(current_action), 'UNK')} (Hold)"
            
            cv2.putText(frame, action_text, (20, 55), font, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, action_text, (20, 55), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            frames.append(frame)

    env.close()

    # 5. Save Video
    print(f"Saving video with {len(frames)} frames...")
    os.makedirs(os.path.dirname(VIDEO_OUTPUT), exist_ok=True)
    imageio.mimsave(VIDEO_OUTPUT, frames, fps=15)
    print(f"Video saved to: {VIDEO_OUTPUT}")

if __name__ == "__main__":
    record_merge_smooth()