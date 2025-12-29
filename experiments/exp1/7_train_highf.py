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

# --- 2. Environment Factory (HIGH FREQUENCY) ---
def make_high_freq_env(rank, model_mode="aggressive", seed=0):
    def _init():
        env_name = "highway-v0"
        
        # BASE CONFIG
        config = {
            "observation": {
                "type": "Kinematics",
                "vehicles_count": 15,
                "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
                "features_range": { "x": [-100, 100], "y": [-100, 100], "vx": [-20, 20], "vy": [-20, 20] },
                "absolute": False, "order": "sorted"
            },
            "action": { "type": "DiscreteMetaAction" },
            "lanes_count": 4,
            "vehicles_count": 50,
            "vehicles_density": 1.5, 
            "duration": 60, 
            "simulation_frequency": 15,
            
            # --- THE GAME CHANGER: 5 Hz REFLEXES ---
            # 5 decisions per second. Reaction time = 0.2s.
            "policy_frequency": 5, 
            # ---------------------------------------
            "normalize_reward": False
        }

        # MODE SPECIFIC REWARDS
        if model_mode == "aggressive":
            # The "Formula 1" Driver
            config.update({
                "collision_reward": -30,    # Serious but not paralyzing
                "right_lane_reward": 0,    
                "high_speed_reward": 0.6,   # High incentive
                "lane_change_reward": 0,    # Precision only (No weaving bonus)
                "reward_speed_range": [15, 35]
            })
        else:
            # The "Attentive Grandma" (High Freq Conservative)
            config.update({
                "collision_reward": -40,    # Fear is dominant
                "right_lane_reward": 0,
                "high_speed_reward": 0.3,   # Low incentive
                "lane_change_reward": -0.1,
                "reward_speed_range": [10, 30]
            })

        env = gym.make(env_name, render_mode=None)
        env.unwrapped.configure(config)
        env.reset(seed=seed + rank)
        return env
    return _init

# --- 3. Training Function ---
def train_high_freq(model_type="QRDQN", mode="aggressive", seed=1000):
    
    num_cpu = max(1, os.cpu_count() - 2)
    print(f"--- Exp 1C (High-Freq {mode}): Using {num_cpu} CPUs... ---")

    # Create Env with 'mode' (conservative or aggressive)
    env = SubprocVecEnv([make_high_freq_env(i, mode, seed) for i in range(num_cpu)])
    env = VecMonitor(env)
    set_random_seed(seed)

    log_dir = os.path.join(script_dir, "logs", f"exp1c_{mode}_highfreq_{model_type}")
    os.makedirs(log_dir, exist_ok=True)

    model_args = dict(
        policy="MlpPolicy",
        env=env,
        policy_kwargs=dict(net_arch=[256, 256]), 
        verbose=0,
        seed=seed,
        tensorboard_log=log_dir,
        device="cpu",
        learning_rate=3e-4,
        buffer_size=300000, 
        learning_starts=5000,
        batch_size=128,
        gamma=0.98,
        target_update_interval=500,
        train_freq=1,
        gradient_steps=1,
        exploration_fraction=0.4,
        exploration_final_eps=0.05
    )

    if model_type == "QRDQN":
        model = QRDQN(**model_args)
    else:
        model = DQN(**model_args)
    
    print(f"Starting {mode.upper()} High-Freq Training - Seed {seed}")
    
    # 5 Hz means we generate 5x more steps per minute.
    # We train for 1M steps to give it enough "real time" experience.
    total_steps = 1000000 
    
    checkpoint_callback = CheckpointCallback(
        save_freq=max(1, 100000 // num_cpu), 
        save_path=os.path.join(script_dir, "models_checkpoints_highfreq"),
        name_prefix=f"{mode}_highfreq_{model_type}"
    )

    model.learn(total_timesteps=total_steps, callback=[CrashLoggingCallback(), checkpoint_callback])
    
    save_path = os.path.join(script_dir, "models", f"exp1c_{mode}_highfreq_{model_type}_s{seed}")
    model.save(save_path)
    print(f"Model saved to {save_path}")
    env.close()

if __name__ == "__main__":
    # RECOMMENDATION: Train the Aggressive QRDQN first.
    # This is your best shot at beating the Density 1.5 wall.
    train_high_freq("DQN", mode="aggressive", seed=200)
    
    # Optional: Train Conservative to compare (if you have time)
    train_high_freq("DQN", mode="conservative", seed=200)