import gymnasium as gym
import highway_env
from stable_baselines3 import DQN
from sb3_contrib import QRDQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
import os

# --- 1. Custom Callback for Safety Metrics ---
class CrashLoggingCallback(BaseCallback):
    """
    Logs 'crashes' specifically. 
    Standard SB3 logs 'reward', but we want to know how many times we hit things.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.crashes = 0

    def _on_step(self) -> bool:
        # Check if the episode ended in a crash
        # highway-env info dict contains 'crashed' boolean
        dones = self.locals['dones']
        infos = self.locals['infos']
        
        for i, done in enumerate(dones):
            if done and infos[i].get("crashed", False):
                self.crashes += 1
                self.logger.record("safety/cumulative_crashes", self.crashes)
        return True

# --- 2. Training Function ---
def train_experiment(model_type="DQN", seed=1):
    env_name = "highway-v0"
    
    # --- FIX IS HERE: Define config dictionary first ---
    env_config = {
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 15,
            "features": ["presence", "x", "y", "vx", "vy"], 
            "features_range": {
                "x": [-100, 100],
                "y": [-100, 100],
                "vx": [-20, 20],
                "vy": [-20, 20]
            },
            "absolute": False
        },
        "action": {
            "type": "DiscreteMetaAction",
        },
        "simulation_frequency": 15,  # [Hz]
        "policy_frequency": 1,       # [Hz]
    }

    # --- FIX IS HERE: Pass 'config' directly to gym.make ---
    env = gym.make(env_name, render_mode=None, config=env_config)
    
    # Wrap for SB3 logging
    env = Monitor(env)

    # Define Model
    policy_kwargs = dict(net_arch=[256, 256])
    
    # Ensure log directory exists
    log_dir = f"./logs/exp1_{model_type}/"
    os.makedirs(log_dir, exist_ok=True)

    if model_type == "DQN":
        model = DQN("MlpPolicy", env, policy_kwargs=policy_kwargs, 
                    verbose=1, seed=seed, tensorboard_log=log_dir)
    elif model_type == "QRDQN":
        model = QRDQN("MlpPolicy", env, policy_kwargs=policy_kwargs, 
                      verbose=1, seed=seed, tensorboard_log=log_dir)
    
    # Train
    print(f"Starting training: {model_type} - Seed {seed}")
    steps = 50000 
    model.learn(total_timesteps=steps, callback=CrashLoggingCallback())
    
    # Save
    # Ensure models directory exists
    save_dir = "../../models"
    os.makedirs(save_dir, exist_ok=True)
    save_path = f"{save_dir}/exp1_{model_type}_s{seed}"
    
    model.save(save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    # Train both models
    train_experiment("DQN", seed=1)
    train_experiment("QRDQN", seed=1)