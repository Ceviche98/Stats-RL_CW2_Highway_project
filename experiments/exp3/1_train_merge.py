import gymnasium as gym
import highway_env
from sb3_contrib import QRDQN
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
import os
import time

# --- SETUP ---
os.environ["SDL_VIDEODRIVER"] = "dummy"
script_dir = os.path.abspath(".")

# --- 1. Custom Callback ---
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
                self.logger.record("safety/cumulative_crashes", self.crashes)
        return True

# --- 2. Environment Factory (MERGE CONFIG) ---
def make_configure_env(rank, seed=0):
    def _init():
        # EXP 3: GENERALIZATION (Train on Merge)
        env_name = "merge-v0"
        
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
            
            # --- MERGE SPECIFIC SETTINGS ---
            # Traffic density on the main highway
            "vehicles_density": 1.5, 
            
            # Short duration because the map is small
            "duration": 20, 
            
            "simulation_frequency": 15,
            "policy_frequency": 1,
            
            # --- AGGRESSIVE REWARDS (Adapted for Merging) ---
            # Penalty: -15 (Aggressive but not suicidal)
            "collision_reward": -15,    
            
            # Right Lane: 0 (No preference, just survive the merge)
            "right_lane_reward": 0,    
            
            # Speed: 0.6 (Must accelerate to match highway speed!)
            "high_speed_reward": 0.7,   
            
            # Lane Change: 0.1 (Encourage the merge action)
            "lane_change_reward": 0.05, 
            
            # Range: [20, 30] m/s -> [72, 108] km/h.
            # We lower the floor to 20 because the ramp starts slow.
            "reward_speed_range": [23, 30],
            
            "normalize_reward": False
        }

        env = gym.make(env_name, render_mode=None)
        env.unwrapped.configure(env_config)
        env.reset(seed=seed + rank)
        return env
    return _init

# --- 3. Training Function ---
def train_experiment_merge(model_type="QRDQN", seed=500):
    
    # 2 cores reserved for overhead
    num_cpu = max(1, os.cpu_count() - 2)
    print(f"--- Exp 3 (Merge): Using {num_cpu} CPUs... ---")

    env = SubprocVecEnv([make_configure_env(i, seed=seed) for i in range(num_cpu)])
    env = VecMonitor(env)
    set_random_seed(seed)

    # Log to 'exp3_merge'
    log_dir = os.path.join(script_dir, "logs", f"exp3_merge_{model_type}")
    os.makedirs(log_dir, exist_ok=True)

    model_args = dict(
        policy="MlpPolicy",
        env=env,
        policy_kwargs=dict(net_arch=[256, 256]), 
        verbose=0,
        seed=seed,
        tensorboard_log=log_dir,
        device="cpu",
        
        learning_rate=5e-4,
        # Smaller buffer is fine for merge (shorter episodes)
        buffer_size=200000, 
        learning_starts=500,
        batch_size=128,
        gamma=0.9,
        target_update_interval=50,
        train_freq=1,
        gradient_steps=1,
        exploration_fraction=0.3,
        exploration_final_eps=0.05
    )

    if model_type == "QRDQN":
        model = QRDQN(**model_args)
    else:
        # Fallback if you want to test DQN later
        model = DQN(**model_args)
    
    print(f"Starting MERGE Training: {model_type} - Seed {seed}")
    
    start_time = time.time()
    
    # Math Fix for Checkpoints
    checkpoint_freq = max(1, 50000 // num_cpu)
    
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq, 
        save_path=os.path.join(script_dir, "models_checkpoints_merge"),
        name_prefix="merge_{}_{}".format(model_type,seed)
    )

    # 600,000 steps is plenty for Merge (short episodes = many episodes)
    total_steps = 600000
    model.learn(
        total_timesteps=total_steps, 
        callback=[CrashLoggingCallback(), checkpoint_callback]
    )
    
    total_time = (time.time() - start_time) / 60
    print(f"Training finished in {total_time:.2f} minutes.")

    save_dir = os.path.join(script_dir, "models")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"exp3_merge_{model_type}_s{seed}")
    
    model.save(save_path)
    print(f"Model saved to {save_path}")
    
    env.close()

if __name__ == "__main__":
    # We only train QRDQN for this (The Expert Model)
    # Seed 500 to keep it totally separate
    #train_experiment_merge("QRDQN", seed=500)
    train_experiment_merge("DQN", seed=500)