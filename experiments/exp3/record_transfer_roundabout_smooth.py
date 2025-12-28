import gymnasium as gym
import highway_env
from sb3_contrib import QRDQN
import os
import imageio
import numpy as np
import cv2

script_dir = os.path.abspath(".")
model_name = "exp3_merge_QRDQN_s500" 
MODEL_PATH = os.path.join(script_dir, "models/{}.zip".format(model_name))
VIDEO_OUTPUT = os.path.join(script_dir, "results/videos/exp3_transfer_roundabout_final.mp4")

def record_transfer_final():
    env = gym.make("roundabout-v0", render_mode="rgb_array")
    
    # 1. Force Merge Config + Smooth Video Setup
    config = {
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
        "policy_frequency": 15, # Smooth video trick
        "duration": 40,
        "screen_width": 600,
        "screen_height": 600,
        "centering_position": [0.5, 0.6],
        "scaling": 5.5 * 1.3
    }
    
    env.unwrapped.configure(config)
    env.reset()

    print(f"Loading MERGE model: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        print("Model not found.")
        return
    model = QRDQN.load(MODEL_PATH, env=env)

    frames = []
    
    # Record only 2 episodes to keep video short but varied
    for ep in range(3):
        print(f"Recording Roundabout Ep {ep+1}...")
        obs, info = env.reset()
        done = False
        truncated = False
        
        step_counter = 0
        current_action = 1 # Idle
        
        while not (done or truncated):
            # --- ACTION REPEAT LOGIC (1 Hz) ---
            if step_counter % 15 == 0:
                current_action, _ = model.predict(obs, deterministic=True)
            
            obs, reward, done, truncated, info = env.step(current_action)
            step_counter += 1
            
            # --- RENDER & OVERLAY ---
            frame = env.render()
            frame = np.ascontiguousarray(frame, dtype=np.uint8)
            
            # 1. Speed Info
            speed_kmh = 0
            try:
                speed_kmh = env.unwrapped.vehicle.speed * 3.6
            except: pass
            
            # 2. Action Info
            action_map = {0: "LANE LEFT", 1: "IDLE", 2: "LANE RIGHT", 3: "FASTER", 4: "SLOWER"}
            act_str = action_map.get(int(current_action), "UNK")
            
            # 3. Draw Overlays
            font = cv2.FONT_HERSHEY_SIMPLEX
            # Speed
            cv2.putText(frame, f"Speed: {speed_kmh:.1f} km/h", (20, 30), font, 0.6, (0,0,0), 3, cv2.LINE_AA)
            cv2.putText(frame, f"Speed: {speed_kmh:.1f} km/h", (20, 30), font, 0.6, (0,255,0), 2, cv2.LINE_AA)
            # Action
            cv2.putText(frame, f"Action: {act_str}", (20, 55), font, 0.6, (0,0,0), 3, cv2.LINE_AA)
            cv2.putText(frame, f"Action: {act_str}", (20, 55), font, 0.6, (255,255,0), 2, cv2.LINE_AA)
            
            frames.append(frame)

    env.close()
    
    os.makedirs(os.path.dirname(VIDEO_OUTPUT), exist_ok=True)
    imageio.mimsave(VIDEO_OUTPUT, frames, fps=15)
    print(f"Final Video saved to {VIDEO_OUTPUT}")

if __name__ == "__main__":
    record_transfer_final()