import gymnasium as gym
import numpy as np
import os
import sys
import highway_env
from stable_baselines3 import DQN
from gymnasium.wrappers import RecordVideo

# --- CONFIGURATION ---
USE_SAFETY_SHIELD = True  # <--- ENABLED: Forces avoidance
DQN_PATH = "../../models/exp1/E_Low_freq_aggr_DQN_s200.zip"
QRDQN_PATH = "../../models/exp1/F_Low_freq_aggr_QRDQN_s200.zip"
VIDEO_DIR = "../../videos/exp4_blind_spot_shielded/"

os.makedirs(VIDEO_DIR, exist_ok=True)

def get_ttc(ego_vehicle, other_vehicle):
    """Calculate Time-To-Collision."""
    dx = other_vehicle.position[0] - ego_vehicle.position[0]
    dv = ego_vehicle.speed - other_vehicle.speed
    if dv <= 0: return 100.0
    ttc = dx / dv
    return ttc if ttc > 0 else 100.0

def run_visual_test(agent, agent_name, num_episodes=3):
    # 1. SETUP ENVIRONMENT
    try:
        env = gym.make("highway-v0", render_mode="rgb_array")
    except gym.error.NameNotFound:
        print("ERROR: highway-v0 not found. Please run: pip install highway-env")
        return

    # MATCH OBSERVATION TO MODEL TRAINING
    env.unwrapped.configure({
        "lanes_count": 3,
        "vehicles_density": 1.2,
        "duration": 40,
        "screen_width": 600,
        "screen_height": 300,
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 15, 
            "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"], 
            "features_range": {
                "x": [-100, 100], "y": [-100, 100], "vx": [-20, 20], "vy": [-20, 20]
            },
            "absolute": False,
            "order": "sorted"
        }
    })

    # 2. WRAP WITH VIDEO RECORDER
    env = RecordVideo(
        env, 
        video_folder=VIDEO_DIR, 
        name_prefix=f"{agent_name}_shield_TRUE",
        episode_trigger=lambda e: True 
    )

    print(f"\n--- Recording Shielded Video for {agent_name} ---")

    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        
        adversary_triggered = False
        adversary_vehicle = None
        
        while not (done or truncated):
            # A. PREDICT ACTION
            if agent is not None:
                action, _ = agent.predict(obs, deterministic=True)
            else:
                action = env.action_space.sample()

            # B. ADVERSARIAL LOGIC
            ego_vehicle = env.unwrapped.vehicle
            
            # Find target in blind spot
            if adversary_vehicle is None:
                for v in env.unwrapped.road.vehicles:
                    if v is ego_vehicle: continue
                    lat_dist = abs(v.position[1] - ego_vehicle.position[1])
                    long_dist = v.position[0] - ego_vehicle.position[0]
                    
                    if 0 < long_dist < 15 and 3 < lat_dist < 5:
                        adversary_vehicle = v
                        break
            
            # Trigger Cut-In (ROBUST TUPLE FIX)
            if adversary_vehicle is not None and not adversary_triggered:
                # Get current lane info
                current_lane = adversary_vehicle.target_lane_index
                
                # Determine direction
                if adversary_vehicle.position[1] > ego_vehicle.position[1]:
                    lane_change = -1 
                else:
                    lane_change = 1

                # Apply change based on type (Tuple vs Int)
                if isinstance(current_lane, tuple):
                    lane_from, lane_to, lane_id = current_lane
                    new_lane_id = max(0, min(2, lane_id + lane_change))
                    adversary_vehicle.target_lane_index = (lane_from, lane_to, new_lane_id)
                else:
                    new_lane = current_lane + lane_change
                    adversary_vehicle.target_lane_index = max(0, min(2, new_lane))

                adversary_triggered = True
                print(f"  > Ep {episode}: Cut-in triggered!")

            # C. SAFETY SHIELD ACTIVATION
            if adversary_triggered and USE_SAFETY_SHIELD:
                ttc = get_ttc(ego_vehicle, adversary_vehicle)
                # If crash is imminent (< 1.5s), OVERRIDE action
                if ttc < 1.5:
                    action = 4 # Emergency Brake (Action 4 in highway-env)
                    # print("!!! SHIELD ACTIVATED !!!") 

            obs, reward, done, truncated, info = env.step(action)
    
    env.close()
    print(f"Shielded videos saved to {VIDEO_DIR}")

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    print(">>> RUNNING SHIELDED TEST (Safety Override ENABLED) <<<")
    
    # 1. TEST DQN
    if not os.path.exists(DQN_PATH):
        print(f"ERROR: Could not find DQN model at: {os.path.abspath(DQN_PATH)}")
    else:
        print(f"Loading DQN from {DQN_PATH}...")
        dqn_model = DQN.load(DQN_PATH)
        run_visual_test(dqn_model, "DQN", num_episodes=3)

    # 2. TEST QR-DQN
    if not os.path.exists(QRDQN_PATH):
        print(f"Warning: Could not find QR-DQN model at: {QRDQN_PATH}")
    else:
        print(f"Loading QR-DQN from {QRDQN_PATH}...")
        try:
            from sb3_contrib import QRDQN
            qrdqn_model = QRDQN.load(QRDQN_PATH)
        except:
            print("  (Loading as standard DQN class...)")
            qrdqn_model = DQN.load(QRDQN_PATH)
            
        run_visual_test(qrdqn_model, "QRDQN", num_episodes=3)
