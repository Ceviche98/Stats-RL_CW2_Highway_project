import gymnasium as gym
import highway_env
from stable_baselines3 import DQN
from sb3_contrib import QRDQN
from gymnasium.wrappers import RecordVideo
import os

# --- CONFIGURATION ---
# Note: Windows paths use backslashes usually, but Python handles forward slashes fine.
MODEL_PATH = "../../models/exp1_QRDQN_s1.zip"  
VIDEO_FOLDER = "../../results/videos/"
ALGO = "QRDQN"  

def record_experiment():
    # --- 1. Define Config Dictionary First ---
    # This must match your training config exactly
    env_config = {
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 15,
            "features": ["presence", "x", "y", "vx", "vy"],
            "features_range": {
                "x": [-100, 100], "y": [-100, 100], "vx": [-20, 20], "vy": [-20, 20]
            },
            "absolute": False
        },
        "simulation_frequency": 15,
        "policy_frequency": 1,
    }

    # --- 2. Pass config inside gym.make() ---
    # This fixes the "OrderEnforcing" error
    env = gym.make("highway-v0", render_mode="rgb_array", config=env_config)

    # 3. Wrap the env to record video
    env = RecordVideo(env, video_folder=VIDEO_FOLDER, 
                      episode_trigger=lambda e: True, 
                      name_prefix=f"visualization_{ALGO}")

    # 4. Load the Model
    print(f"Loading {ALGO} from {MODEL_PATH}...")
    
    # Check if file exists before loading to avoid confusion
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model file not found at {MODEL_PATH}")
        print("Did you finish training yet?")
        return

    if ALGO == "DQN":
        model = DQN.load(MODEL_PATH, env=env)
    elif ALGO == "QRDQN":
        model = QRDQN.load(MODEL_PATH, env=env)

    # 5. Run the Simulation
    obs, info = env.reset()
    done = False
    truncated = False
    
    print("Recording episode...")
    while not (done or truncated):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        env.render()
        
    env.close()
    print(f"Video saved to {VIDEO_FOLDER}")

if __name__ == "__main__":
    os.makedirs(VIDEO_FOLDER, exist_ok=True)
    record_experiment()