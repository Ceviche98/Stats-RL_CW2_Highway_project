import gymnasium as gym
from gymnasium.wrappers import TransformObservation
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from sb3_contrib import QRDQN
import os

# --- WRAPPERS FOR NOISE ---
class SensorNoiseWrapper(gym.ObservationWrapper):
    def __init__(self, env, noise_std=0.0):
        super().__init__(env)
        self.noise_std = noise_std
    
    def observation(self, obs):
        # Add Gaussian noise to coordinates [x, y, vx, vy]
        noise = np.random.normal(0, self.noise_std, size=obs.shape)
        return obs + noise

# --- CONFIG ---
MODELS = {
    "DQN": "../../models/exp1_DQN_s1.zip",
    "QR-DQN": "../../models/exp1_QRDQN_s1.zip"
}
NOISE_LEVELS = [0.0, 0.1, 0.3, 0.5] # 0 = Clean, 0.5 = High Chaos
RESULTS_DIR = "../../results/exp2_robustness/"
os.makedirs(RESULTS_DIR, exist_ok=True)

def evaluate_robustness(model_path, algo_name, noise_level):
    # Create env WITH NOISE wrapper
    env = gym.make("highway-v0", render_mode=None)
    env = SensorNoiseWrapper(env, noise_std=noise_level)
    
    if algo_name == "DQN":
        model = DQN.load(model_path, env=env)
    else:
        model = QRDQN.load(model_path, env=env)
        
    crashes = 0
    episodes = 20 # Keep it small for quick checks
    
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, truncated, info = env.step(action)
            done = done or truncated
            if info.get('crashed', False):
                crashes += 1
                break # Count crash and restart
    
    return crashes / episodes

if __name__ == "__main__":
    data = []
    
    for name, path in MODELS.items():
        if not os.path.exists(path): continue
        print(f"Testing {name} robustness...")
        
        for noise in NOISE_LEVELS:
            crash_rate = evaluate_robustness(path, name, noise)
            data.append({
                "Model": name,
                "Noise Std Dev": noise,
                "Crash Rate": crash_rate
            })
            print(f"  > Noise {noise}: Crash Rate {crash_rate:.2f}")

    # Plotting
    df = pd.DataFrame(data)
    plt.figure(figsize=(8, 5))
    sns.lineplot(data=df, x="Noise Std Dev", y="Crash Rate", hue="Model", marker="o")
    plt.title("Robustness Analysis: Crash Rate vs Sensor Noise")
    plt.ylabel("Crash Probability")
    plt.grid(True)
    plt.savefig(f"{RESULTS_DIR}/robustness_curve.png")
    print("Robustness plot saved!")