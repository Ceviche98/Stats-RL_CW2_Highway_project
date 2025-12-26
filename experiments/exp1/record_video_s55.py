import gymnasium as gym
import highway_env
from sb3_contrib import QRDQN
from gymnasium.wrappers import RecordVideo
import os

# --- CONFIGURATION ---
script_dir = os.path.abspath(".")

# 1. EXACT MODEL MATCHING
# We use s55 as you specified
model_name = "exp1_QRDQN_s55" 
MODEL_PATH = os.path.join(script_dir, "models/{}.zip".format(model_name))
VIDEO_FOLDER = os.path.join(script_dir, "results/videos/")

def record_realtime_video():
    # Define the exact config used in training
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
            "vehicles_density": 1.75, 
            "duration": 40, 
            "simulation_frequency": 15,
            "policy_frequency": 1,
            
            # These don't affect recording, but good for consistency
            "collision_reward": -20,    
            "right_lane_reward": 0,    
            "high_speed_reward": 0.4,   
            "lane_change_reward": 0, 
            "reward_speed_range": [20, 30],
            "normalize_reward": False
        }

    # 2. CREATE ENV
    # render_mode must be 'rgb_array' for saving to file
    env = gym.make("highway-v0", render_mode="rgb_array")
    env.unwrapped.configure(env_config)
    env.reset()

    # --- THE CRITICAL FIX ---
    # Since we have 1 step per second (policy_frequency=1), 
    # we must set FPS to 1 to match real-time playback.
    # 40 steps will now take exactly 40 seconds to play.
    env.metadata["render_fps"] = 1

    # Wrap the env
    env = RecordVideo(env, video_folder=VIDEO_FOLDER, 
                      episode_trigger=lambda e: True, 
                      name_prefix=f"realtime_{model_name}")

    # 3. LOAD MODEL
    print(f"Loading QRDQN from {MODEL_PATH}...")
    if not os.path.exists(MODEL_PATH):
        print(f"CRITICAL ERROR: Model file not found at {MODEL_PATH}")
        return

    model = QRDQN.load(MODEL_PATH, env=env)

    # 4. RUN RECORDING
    print("Recording 1 Episode (Should be approx 35-40 seconds long)...")
    
    obs, info = env.reset()
    done = False
    truncated = False
    step = 0
    
    while not (done or truncated):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        step += 1
        
    print(f"Episode finished. Total Steps: {step}")
    print(f"If steps were ~35, video duration will be ~35 seconds.")

    env.close()
    print(f"Video saved to {VIDEO_FOLDER}")

if __name__ == "__main__":
    os.makedirs(VIDEO_FOLDER, exist_ok=True)
    record_realtime_video()