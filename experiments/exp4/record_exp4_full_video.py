import os
import sys
import gymnasium as gym

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from stable_baselines3 import DQN
from sb3_contrib import QRDQN
from gymnasium.wrappers import RecordVideo

from experiments.exp4.cutin_wrapper import BlindSpotCutInWrapper


VIDEO_DIR = "../../videos/exp4_full"
os.makedirs(VIDEO_DIR, exist_ok=True)

MODELS = {
    "DQN": "../../models/exp1_aggressive_DQN_s200.zip",
    "QRDQN": "../../models/exp1_aggressive_QRDQN_s200.zip",
}


def make_env(seed=0):
    env_config = {
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 15,  # match your model training
            "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],  # match training
            "absolute": False,
        },
        "action": {"type": "DiscreteMetaAction"},
        "lanes_count": 4,
        "duration": 120,
        "simulation_frequency": 15,
        "policy_frequency": 1,
    }

    env = gym.make("highway-v0", render_mode="rgb_array", config=env_config)

    env = BlindSpotCutInWrapper(
        env,
        cutin_trigger_distance_m=5.0,
        spawn_ahead_m=20.0,
        npc_speed_delta=-1.0,
        brake_action_index=4,
        verbose=False,
    )

    return env



def record_full_episode(model_name, model_path, seed=0):
    print(f"Recording FULL episode for {model_name}")

    env = make_env(seed)

    env = RecordVideo(
        env,
        video_folder=VIDEO_DIR,
        name_prefix=f"exp4_full_{model_name.lower()}",
        episode_trigger=lambda episode_id: True,  # record entire episode
        disable_logger=True,
    )

    if model_name == "DQN":
        model = DQN.load(model_path, env=env)
    else:
        model = QRDQN.load(model_path, env=env)

    obs, _ = env.reset(seed=seed)
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    env.close()
    print(f"Saved FULL video for {model_name}")


if __name__ == "__main__":
    for name, path in MODELS.items():
        if not os.path.exists(path):
            print(f"Model not found: {path}")
            continue
        record_full_episode(name, path, seed=123)

    print("Full videos saved to:", VIDEO_DIR)
