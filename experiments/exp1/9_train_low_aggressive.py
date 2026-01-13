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
        self.episodes = 0
        self.episode_crashes = []

    def _on_step(self) -> bool:
        # In vectorized envs, 'dones' and 'infos' are lists from all envs
        dones = self.locals['dones']
        infos = self.locals['infos']
        
        for i, done in enumerate(dones):
            # Check if ANY of the parallel envs crashed
            # highway-env puts "crashed": True in info when a collision causes termination
            if done:
                self.episodes += 1
                if infos[i].get("crashed", False):
                    self.crashes += 1
                    self.episode_crashes.append(1)
                else:
                    self.episode_crashes.append(0)
                
                if len(self.episode_crashes) >= 100:
                    recent_crashes = sum(self.episode_crashes[-100:])
                    self.logger.record("safety/crash_rate_last_100", recent_crashes)
                
                self.logger.record("safety/cumulative_crashes", self.crashes)
                self.logger.record("safety/total_episodes", self.episodes)
                if self.episodes > 0:
                    self.logger.record("safety/overall_crash_rate", 
                                      100 * self.crashes / self.episodes)
        return True

# --- 2. Environment Factory ---
def make_low_freq_env(rank, model_mode="aggressive", seed=0):
    """
    Low-frequency environment factory (1 Hz policy frequency).
    Supports both aggressive and conservative modes.
    """
    def _init():
        env_name = "highway-v0"
        
        # --- BASE CONFIGURATION ---
        config = {
            "observation": {
                "type": "Kinematics",
                "vehicles_count": 15,
                "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
                "features_range": {
                    "x": [-150, 150],  # EXPANDED observation range
                    "y": [-100, 100], 
                    "vx": [-20, 20], 
                    "vy": [-20, 20]
                },
                "absolute": False,
                "order": "sorted",
                "see_behind": True      # Critical for lane changes
            },
            "action": { "type": "DiscreteMetaAction" },
            "lanes_count": 4,
            "vehicles_count": 50,
            "vehicles_density": 1.5,    # Expert difficulty
            "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
            "duration": 60, 
            "simulation_frequency": 15,
            "policy_frequency": 2,      # Low frequency: 2 Hz (0.5s decisions)
            "normalize_reward": False
        }

        # --- MODE SPECIFIC REWARDS ---
        if model_mode == "aggressive":
            # Aggressive driver: High speed, lower penalty for crashes
            config.update({
                "collision_reward": -15,    # Baseline collision penalty
                "right_lane_reward": 0,     # No right lane preference
                "high_speed_reward": 0.5,   # Encourage speed
                "lane_change_reward": 0.1,  # Small reward for active driving
                "reward_speed_range": [25, 35]
            })
        else:
            # Conservative driver: Lower speed, higher crash penalty
            config.update({
                "collision_reward": -40,    # Strong penalty
                "right_lane_reward": 0,     # No right lane preference
                "high_speed_reward": 0.3,   # Lower speed encouragement
                "lane_change_reward": -0.1, # Discourage lane changes
                "reward_speed_range": [10, 30]
            })

        # 1. Create Env
        env = gym.make(env_name, render_mode=None)
        
        # 2. Force Config Update
        env.unwrapped.configure(config)
        
        # 3. Reset
        env.reset(seed=seed + rank)
        return env
    return _init

# --- 3. Training Function ---
def train_low_freq(model_type="QRDQN", mode="aggressive", seed=300, model_index="K", 
                   gamma=0.93, collision_reward=-13, policy_freq=2, obs_range_x=[-150, 150]):
    """
    Train low-frequency model with specified algorithm and hyperparameters.
    
    Args:
        model_type: "DQN" or "QRDQN"
        mode: "aggressive" or "conservative"
        seed: Random seed for reproducibility
        model_index: Index label for model naming (K, L, M, etc.)
        gamma: Discount factor
        collision_reward: Collision penalty
        policy_freq: Policy frequency (Hz)
        obs_range_x: Observation range for x coordinate
    """
    
    num_cpu = max(1, os.cpu_count() - 2)
    print(f"\n--- Row {model_index}: Low-Freq {mode.upper()} ({model_type}), γ={gamma}: Using {num_cpu} CPUs... ---")

    # Create the vectorized environment with custom parameters
    def make_custom_env(rank):
        def _init():
            env_name = "highway-v0"
            
            config = {
                "observation": {
                    "type": "Kinematics",
                    "vehicles_count": 15,
                    "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
                    "features_range": {
                        "x": obs_range_x,  # Custom observation range
                        "y": [-100, 100], 
                        "vx": [-20, 20], 
                        "vy": [-20, 20]
                    },
                    "absolute": False,
                    "order": "sorted",
                    "see_behind": True
                },
                "action": { "type": "DiscreteMetaAction" },
                "lanes_count": 4,
                "vehicles_count": 50,
                "vehicles_density": 1.5,
                "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
                "duration": 60, 
                "simulation_frequency": 15,
                "policy_frequency": policy_freq,  # Custom policy frequency
                "normalize_reward": False
            }

            # MODE SPECIFIC REWARDS
            if mode == "aggressive":
                config.update({
                    "collision_reward": collision_reward,  # Custom collision penalty
                    "right_lane_reward": 0,
                    "high_speed_reward": 0.5,
                    "lane_change_reward": 0.1,
                    "reward_speed_range": [25, 35]
                })
            else:
                config.update({
                    "collision_reward": -40,
                    "right_lane_reward": 0,
                    "high_speed_reward": 0.3,
                    "lane_change_reward": -0.1,
                    "reward_speed_range": [10, 30]
                })

            env = gym.make(env_name, render_mode=None)
            env.unwrapped.configure(config)
            env.reset(seed=seed + rank)
            return env
        return _init

    env = SubprocVecEnv([make_custom_env(i) for i in range(num_cpu)])
    env = VecMonitor(env)
    set_random_seed(seed)

    # --- PATHS ---
    log_dir = os.path.join(script_dir, "logs", f"exp1_row{model_index}_{model_type}")
    os.makedirs(log_dir, exist_ok=True)

    # --- CONFIGURATION ---
    model_args = dict(
        policy="MlpPolicy",
        env=env,
        policy_kwargs=dict(net_arch=[256, 256]), 
        verbose=0,
        seed=seed,
        tensorboard_log=log_dir,
        device="cpu",
        
        # Learning hyperparameters
        learning_rate=5e-4,
        buffer_size=500000,
        learning_starts=20000,
        batch_size=128,
        gamma=gamma,  # Custom gamma
        target_update_interval=1000,
        train_freq=4,
        gradient_steps=4,
        exploration_fraction=0.4,
        exploration_final_eps=0.1
    )

    if model_type == "QRDQN":
        model = QRDQN(**model_args)
    else:
        raise ValueError(f"Row {model_index} requires QRDQN")
    
    print(f"Starting Row {model_index} Training - Seed {seed}")
    print(f"Config: γ={gamma}, R_col={collision_reward}, Freq={policy_freq}Hz, x∈{obs_range_x}")
    
    start_time = time.time()
    
    # Checkpoint periodically
    checkpoint_callback = CheckpointCallback(
        save_freq=max(1, 250000 // num_cpu), 
        save_path=os.path.join(script_dir, "models_checkpoints"),
        name_prefix=f"row{model_index}_{model_type}"
    )

    total_steps = 3_000_000
    model.learn(
        total_timesteps=total_steps, 
        callback=[CrashLoggingCallback(), checkpoint_callback]
    )
    
    total_time = (time.time() - start_time) / 60
    print(f"Training finished in {total_time:.2f} minutes.")

    # Save Final Model
    save_dir = os.path.join(script_dir, "models", "exp1")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{model_index}_Low_freq_{mode}_{model_type}_s{seed}_opt")
    
    model.save(save_path)
    print(f"Model saved to {save_path}")
    
    # Close envs to free RAM
    env.close()

if __name__ == "__main__":
    # Train models K, L, M from TABLE III
    # All are Low-Freq, Aggressive, QRDQN with variations
    
    # Row K: γ: 0.93, R_col: -13, Obs: see_behind + x ∈ [-150, 150]
    train_low_freq(model_type="QRDQN", mode="aggressive", seed=300, model_index="K",
                   gamma=0.93, collision_reward=-13, policy_freq=2, obs_range_x=[-150, 150])
    
    # Row L: Freq: 2 Hz, Params same as K (baseline for 2 Hz)
    train_low_freq(model_type="QRDQN", mode="aggressive", seed=300, model_index="L",
                   gamma=0.93, collision_reward=-13, policy_freq=2, obs_range_x=[-150, 150])
    
    # Row M: Obs: Baseline + see_behind + x ∈ [-150, 150]
    train_low_freq(model_type="QRDQN", mode="aggressive", seed=300, model_index="M",
                   gamma=0.93, collision_reward=-13, policy_freq=2, obs_range_x=[-150, 150])