import gymnasium as gym
import highway_env
from sb3_contrib import QRDQN
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
import os

# --- SETUP ---
script_dir = os.path.abspath(".")

# Define your models (Ensure names match your saved files)
MODELS = [
    {"name": "DQN",   "class": DQN,   "file": "exp3_merge_DQN_s500"},
    {"name": "QRDQN", "class": QRDQN, "file": "exp3_merge_QRDQN_s500"}
]

class CrashLoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.crashes = 0

    def _on_step(self) -> bool:
        dones = self.locals['dones']
        infos = self.locals['infos']
        for i, done in enumerate(dones):
            if done and infos[i].get("crashed", False):
                self.crashes += 1
                self.logger.record("fine_tune/cumulative_crashes", self.crashes)
        return True

def make_roundabout_env(rank):
    def _init():
        env = gym.make("roundabout-v0", render_mode=None)
        # FORCE MERGE CONFIG
        config = {
            "observation": {
                "type": "Kinematics",
                "vehicles_count": 15,
                "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
                "features_range": { "x": [-100, 100], "y": [-100, 100], "vx": [-20, 20], "vy": [-20, 20] },
                "absolute": False, "order": "sorted"
            },
            "action": { "type": "DiscreteMetaAction" },
            "simulation_frequency": 15, "policy_frequency": 1,
            "duration": 40,
            "vehicles_density": 1.0
        }
        env.unwrapped.configure(config)
        env.reset()
        return env
    return _init

def run_finetuning():
    for model_cfg in MODELS:
        print(f"--- Fine-Tuning {model_cfg['name']} ---")
        
        # 1. Setup Env
        env = SubprocVecEnv([make_roundabout_env(i) for i in range(4)])
        env = VecMonitor(env)
        
        # 2. Load Model
        path = os.path.join(script_dir, "models/{}.zip".format(model_cfg['file']))
        if not os.path.exists(path):
            print(f"Skipping {model_cfg['name']}, file not found.")
            continue
            
        model = model_cfg['class'].load(path, env=env)
        
        # 3. Setup Logging
        log_dir = os.path.join(script_dir, "logs", f"exp3_finetune_{model_cfg['name']}")
        model.set_logger(None)
        model.tensorboard_log = log_dir
        
        # 4. Train for 50k steps (Short Adaptation Phase)
        model.learn(total_timesteps=50000, callback=CrashLoggingCallback())
        
        # 5. Save
        save_path = os.path.join(script_dir, f"models/exp3_finetuned_{model_cfg['name']}")
        model.save(save_path)
        print(f"Saved fine-tuned {model_cfg['name']} to {save_path}")
        
        env.close()

if __name__ == "__main__":
    run_finetuning()