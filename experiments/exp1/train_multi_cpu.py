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
    :param rank: index of the subprocess (0, 1, 2...)
    :param seed: the initial seed for RNG
    """
    def _init():
        env_name = "highway-v0"
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
            "action": { "type": "DiscreteMetaAction" },
            "simulation_frequency": 15,
            "policy_frequency": 1,
            "render_mode": None,
            "lanes_count": 4,
        }
        env = gym.make(env_name, render_mode=None, config=env_config)
        # Important: Seed each environment differently so they don't copy each other!
        env.reset(seed=seed + rank)
        return env
    return _init

# --- 3. Training Function ---
def train_experiment(model_type="DQN", seed=1):
    
    # Detect CPU cores (Leave 1 free for system if you want, or use all)
    # If you have 8 logical cores, this will run 8x faster per step.
    num_cpu = os.cpu_count()  
    print(f"--- Detected {num_cpu} CPUs. Launching parallel environments... ---")

    # Create the vectorized environment
    # SubprocVecEnv = Run in separate processes (True parallelism)
    env = SubprocVecEnv([make_configure_env(i, seed=seed) for i in range(num_cpu)])
    
    # Wrap in VecMonitor to log rewards/stats for the whole batch
    env = VecMonitor(env)
    
    # Set seeds for the algo
    set_random_seed(seed)

    # --- MODEL PARAMETERS ---
    policy_kwargs = dict(net_arch=[256, 256])
    
    log_dir = os.path.join(script_dir, "logs", f"exp1_{model_type}")
    os.makedirs(log_dir, exist_ok=True)

    # Common args to ensure FAIR comparison
    model_args = dict(
        policy="MlpPolicy",
        env=env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=seed,
        tensorboard_log=log_dir,
        device="cpu",
        learning_rate=5e-4,
        buffer_size=50000,
        learning_starts=1000,
        batch_size=128,
        tau=1.0,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=2000,
        exploration_fraction=0.3,
        exploration_final_eps=0.05
    )

    if model_type == "DQN":
        model = DQN(**model_args)
    elif model_type == "QRDQN":
        model = QRDQN(**model_args)
    
    print(f"Starting training: {model_type} on {num_cpu} cores")
    start_time = time.time()
    
    # 200,000 steps distributed across N CPUs
    # If N=8, each CPU only does 25,000 steps!
    total_steps = 200000
    model.learn(total_timesteps=total_steps, callback=CrashLoggingCallback())
    
    total_time = (time.time() - start_time) / 60
    print(f"Training finished in {total_time:.2f} minutes.")

    # Save
    save_dir = os.path.join(script_dir, "models")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"exp1_{model_type}_s{seed}")
    model.save(save_path)
    print(f"Model saved to {save_path}")
    
    # Close the parallel processes
    env.close()

if __name__ == "__main__":
    # Windows/Multiprocessing requires this protection block
    train_experiment("DQN", seed=42)
    train_experiment("QRDQN", seed=42)