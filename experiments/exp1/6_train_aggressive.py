import gymnasium as gym
import highway_env
from stable_baselines3 import DQN
from sb3_contrib import QRDQN
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
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
            # highway-env puts "crashed": True in info when a collision causes termination
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
            "vehicles_density": 1.5, # Expert difficulty
            "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
            "duration": 60, 
            "simulation_frequency": 15,
            "policy_frequency": 1,
            
            # --- OPTIMIZED REWARDS (User: Gamma 0.9 Specs) ---
            "collision_reward": -15,    # The "Death Sentence"
            "right_lane_reward": 0,     # Encourages overtaking
            "high_speed_reward": 0.5,   # Overshadowed by crash penalty
            "lane_change_reward": 0.1, 
            "reward_speed_range": [25, 35],
            "normalize_reward": False
        }

        # 1. Create Env
        env = gym.make(env_name, render_mode=None)
        
        # 2. FORCE CONFIG UPDATE
        env.unwrapped.configure(env_config)
        
        # 3. Reset
        env.reset(seed=seed + rank)
        return env
    return _init

# --- 3. Training Function ---
def train_experiment(model_type="DQN", seed=1):
    
    # SAFEGUARD: Don't spawn 32 processes on a workstation. Cap at 8 or 10.
    # If you have a specific server, you can increase this to 16.
    #max_cpus = 16 
    total_cores = os.cpu_count()
    num_cpu = max(1, total_cores - 2)
    model_index= "E" if model_type=="DQN" else "F"
    print(f"--- Detected {os.cpu_count()} CPUs. Using {num_cpu} parallel environments... ---")

    # Create the vectorized environment
    env = SubprocVecEnv([make_configure_env(i, seed=seed) for i in range(num_cpu)])
    
    # Wrap in VecMonitor to log rewards/stats for the whole batch
    env = VecMonitor(env)
    
    # Set seeds
    set_random_seed(seed)

    # --- PATHS ---
    log_dir = os.path.join(script_dir, "logs", f"exp1_{model_type}_aggressive")
    os.makedirs(log_dir, exist_ok=True)

    # --- HYBRID CONFIGURATION (FINAL) ---
    model_args = dict(
        policy="MlpPolicy",
        env=env,
        policy_kwargs=dict(net_arch=[256, 256]), 
        verbose=0,
        seed=seed,
        tensorboard_log=log_dir,
        device="cpu",
        
        # 1. Learning Rate
        learning_rate=5e-4,
        
        # 2. Buffer: 200k (PREVENTS CATASTROPHIC FORGETTING)
        buffer_size=350000,
        
        # 3. Learning Starts: 500
        learning_starts=500,
        
        # 4. Batch Size: 128 (STABILIZES -20 PENALTY)
        batch_size=128,
        
        # 5. Gamma: 0.9 (LOOKS 10 STEPS AHEAD)
        gamma=0.9,
        
        # 6. Target Update
        target_update_interval=50,
        
        train_freq=1,
        gradient_steps=1,
        
        exploration_fraction=0.3,
        exploration_final_eps=0.05
    )

    if model_type == "DQN":
        model = DQN(**model_args)
    elif model_type == "QRDQN":
        model = QRDQN(**model_args)
    
    print(f"Starting training: {model_type} - Seed {seed}")
    print(f"Config: Gamma={model_args['gamma']}, Batch={model_args['batch_size']}, Buffer={model_args['buffer_size']}")
    
    start_time = time.time()
    
    # Checkpoint every 50k steps
    checkpoint_callback = CheckpointCallback(
        save_freq=50000, 
        save_path=os.path.join(script_dir, "models_checkpoints"),
        name_prefix="{}_checkpoint_aggr_{}".format(model_type,seed)
    )

    total_steps = 750000
    model.learn(
        total_timesteps=total_steps, 
        callback=[CrashLoggingCallback(), checkpoint_callback]
    )
    
    total_time = (time.time() - start_time) / 60
    print(f"Training finished in {total_time:.2f} minutes.")

    # Save Final Model
    save_dir = os.path.join(script_dir, "models/exp1/")
    os.makedirs(save_dir, exist_ok=True)
    # Assign model index based on model type (E=DQN, F=QRDQN)
    model_idx = "E" if model_type == "DQN" else "F"
    save_path = os.path.join(save_dir, f"{model_idx}_Low_freq_aggr_{model_type}_s{seed}")
    
    model.save(save_path)
    print(f"Model saved to {save_path}")
    
    # Close envs to free RAM
    env.close()

if __name__ == "__main__":
    # Train both
    train_experiment("QRDQN", seed=200)
    train_experiment("DQN", seed=200)