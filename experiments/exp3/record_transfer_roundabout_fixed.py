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
VIDEO_OUTPUT = os.path.join(script_dir, "results/videos/exp3_transfer_roundabout_fixed.mp4")

def record_transfer_fixed():
    env = gym.make("roundabout-v0", render_mode="rgb_array")
    
    # Config matching your training
    config = {
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 15, 
            "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
            "features_range": { "x": [-100, 100], "y": [-100, 100], "vx": [-20, 20], "vy": [-20, 20] },
            "absolute": False, "order": "sorted"
        },
        "action": { "type": "DiscreteMetaAction" },
        "simulation_frequency": 15,
        "policy_frequency": 15, 
        "duration": 40,
        "screen_width": 600, "screen_height": 600,
        "centering_position": [0.5, 0.6], "scaling": 5.5 * 1.3
    }
    
    env.unwrapped.configure(config)
    env.reset()

    if not os.path.exists(MODEL_PATH):
        print("Model not found.")
        return
    model = QRDQN.load(MODEL_PATH, env=env)

    frames = []
    
    # Record just 1 perfect episode
    obs, info = env.reset()
    done = False
    truncated = False
    step_counter = 0
    current_action = 1
    
    print("Recording...")
    while not (done or truncated):
        # 1 Hz Action Logic
        if step_counter % 15 == 0:
            current_action, _ = model.predict(obs, deterministic=True)
        
        obs, reward, done, truncated, info = env.step(current_action)
        step_counter += 1
        
        # --- SUCCESS CHECK (STOP RECORDING IF EXIT REACHED) ---
        # In Roundabout, x > 0 and y > 0 usually means we successfully took the exit
        # Adjust these values based on visual feedback if needed
        x_pos = env.unwrapped.vehicle.position[0]
        if x_pos > 80: # Car has left the screen to the right
            print("Car exited successfully. Stopping video.")
            break
        
        # --- RENDER OVERLAYS ---
        frame = env.render()
        frame = np.ascontiguousarray(frame, dtype=np.uint8)
        
        speed_kmh = 0
        try: speed_kmh = env.unwrapped.vehicle.speed * 3.6
        except: pass
        
        action_map = {0: "LANE LEFT", 1: "IDLE", 2: "LANE RIGHT", 3: "FASTER", 4: "SLOWER"}
        act_str = action_map.get(int(current_action), "UNK")
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, f"Speed: {speed_kmh:.1f} km/h", (20, 30), font, 0.6, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(frame, f"Speed: {speed_kmh:.1f} km/h", (20, 30), font, 0.6, (0,255,0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Action: {act_str}", (20, 55), font, 0.6, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(frame, f"Action: {act_str}", (20, 55), font, 0.6, (255,255,0), 2, cv2.LINE_AA)
        
        frames.append(frame)

    env.close()
    os.makedirs(os.path.dirname(VIDEO_OUTPUT), exist_ok=True)
    imageio.mimsave(VIDEO_OUTPUT, frames, fps=15)
    print(f"Video saved to {VIDEO_OUTPUT}")

if __name__ == "__main__":
    record_transfer_fixed()