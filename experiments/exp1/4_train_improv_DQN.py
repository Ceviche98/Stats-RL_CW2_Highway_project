import gymnasium as gym
import highway_env
from stable_baselines3 import DQN
from sb3_contrib import QRDQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
import os
import time

# --- SETUP ---
os.environ["SDL_VIDEODRIVER"] = "dummy"
script_dir = os.path.abspath(".")

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
        # In vectorized envs, 'dones' and 'infos' are lists from all envs
        dones = self.locals['dones']
        infos = self.locals['infos']
        
        for i, done in enumerate(dones):
            # Check if ANY of the parallel envs crashed
            if done and infos[i].get("crashed", False):
                self.crashes += 1
                self.logger.record("safety/cumulative_crashes", self.crashes)
        return True

# --- 2. Environment Factory ---
def make_configure_env(rank, seed=0):
    """
    Utility function for multiprocessed env.
    Includes the 'Force Config' fix to ensure traffic density is applied.
    """
    def _init():
        env_name = "highway-v0"
        
        # --- RESEARCH GRADE CONFIGURATION ---
        # Based on Highway-Env Official Docs
        env_config = {
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 15, 
            # Added cos_h and sin_h for better prediction of lane changes
            "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"], 
            "features_range": {
                "x": [-100, 100], "y": [-100, 100], "vx": [-20, 20], "vy": [-20, 20]
            },
            "absolute": False,
            "order": "sorted"
        },
        "action": { "type": "DiscreteMetaAction" },
        "lanes_count": 4,
        
        # --- TRAFFIC ---
        "vehicles_count": 50,
        "vehicles_density": 1.75,
        # Explicitly defining NPC behavior ensures consistency
        "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
        "duration": 40,
        
        # --- PHYSICS ---
        "simulation_frequency": 15,
        "policy_frequency": 1,
        
        # --- REWARDS ---
        "collision_reward": -1,    
        "right_lane_reward": 0,  
        "high_speed_reward": 0.8,  
        "lane_change_reward": 0, # Small incentive to be dynamic
        "reward_speed_range": [20, 30],
        "normalize_reward": True
        }

        # 1. Create Env
        env = gym.make(env_name, render_mode=None)
        
        # 2. FORCE CONFIG UPDATE (The Fix)
        # Bypasses gym.make() sometimes ignoring config
        env.unwrapped.configure(env_config)
        
        # 3. Reset
        env.reset(seed=seed + rank)
        return env
    return _init

# --- 3. Training Function ---
def train_experiment(model_type="DQN", seed=1):
    
    # Detect CPU cores
    num_cpu = os.cpu_count()  
    #num_cpu = 8
    print(f"--- Detected {num_cpu} CPUs. Launching parallel environments... ---")

    # Create the vectorized environment
    env = SubprocVecEnv([make_configure_env(i, seed=seed) for i in range(num_cpu)])
    
    # Wrap in VecMonitor to log rewards/stats for the whole batch
    env = VecMonitor(env)
    
    # Set seeds
    set_random_seed(seed)

    # --- MODEL PARAMETERS ---
    policy_kwargs = dict(net_arch=[256, 256])
    
    log_dir = os.path.join(script_dir, "logs", f"exp1_{model_type}")
    os.makedirs(log_dir, exist_ok=True)

    # --- HYBRID CONFIGURATION ---
    # Combines 'Repo' speed (TargetUpdate=50) with 'Paper' stability (Batch=32)
    model_args = dict(
        policy="MlpPolicy",
        env=env,
        policy_kwargs=policy_kwargs,
        verbose=0,
        seed=seed,
        tensorboard_log=log_dir,
        device="cpu",
        
        # 1. Learning Rate: Aggressive enough for the physics, but not 5e-5 (too slow)
        learning_rate=5e-4,
        
        # 2. Buffer: Smaller buffer keeps data "fresh" relevant to current policy
        buffer_size=15000,
        
        # 3. Learning Starts: Start fast. No need to wait 50k steps.
        learning_starts=200,
        
        # 4. Batch Size: 32 updates weights more frequently than 128
        batch_size=32,
        
        # 5. Gamma: 0.8 makes agent prioritize IMMEDIATE survival over infinite future
        gamma=0.8,
        
        # 6. Target Update: 50 is the 'Magic Number' for highway-env stability
        target_update_interval=50,
        
        train_freq=1,
        gradient_steps=1,
        
        # 7. Exploration: 30% exploration gives it time to learn safe paths
        exploration_fraction=0.3,
        exploration_final_eps=0.05
    )

    if model_type == "DQN":
        model = DQN(**model_args)
    elif model_type == "QRDQN":
        model = QRDQN(**model_args)
    
    print(f"Starting training: {model_type} - Seed {seed}")
    start_time = time.time()
    
    # 200k steps is plenty with this setup
    total_steps = 500000
    model.learn(total_timesteps=total_steps, callback=CrashLoggingCallback())
    
    total_time = (time.time() - start_time) / 60
    print(f"Training finished in {total_time:.2f} minutes.")

    # Save
    save_dir = os.path.join(script_dir, "models")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"exp1_{model_type}_s{seed}")
    
    model.save(save_path)
    print(f"Model saved to {save_path}")
    
    # Close envs
    env.close()

if __name__ == "__main__":
    # Train both
    train_experiment("QRDQN", seed=50)
    train_experiment("DQN", seed=50)
    