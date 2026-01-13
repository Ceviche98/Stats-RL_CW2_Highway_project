import gymnasium as gym
import highway_env
from stable_baselines3 import DQN
from sb3_contrib import QRDQN
from stable_baselines3.common.vec_env import SubprocVecEnv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import warnings
from tqdm import tqdm  # <--- Importamos la barra de carga

# --- 1. SILENCIAR WARNINGS ---
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Silencia logs de bajo nivel

# --- CONFIGURATION ---
N_EPISODES = 1000   
N_CPU = 12          # Ryzen 9900X Cores
RESULTS_DIR = "../../results/exp1/"
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- INPUT DICTIONARY ---
MODELS = {
    "High_Frequency": {
        "QRDQN": {
            "Aggressive": "workspace/models/exp1/B_High_freq_aggressive_QRDQN_s200.zip",
            "Conservative": "workspace/models/exp1/D_High_freq_conservative_QRDQN_s200.zip"
        },
        "DQN": {
            "Aggressive": "workspace/models/exp1/A_High_freq_aggressive_DQN_s200.zip",
            "Conservative": "workspace/models/exp1/C_High_freq_conservative_DQN_s200.zip"
        },
    },
    "Low_Frequency": {
        "QRDQN": {
            "Aggressive": "workspace/models/exp1/E_Low_freq_aggr_QRDQN_s200.zip",
            "Conservative": "workspace/models/exp1/G_Low_freq_cons_QRDQN_s200.zip"
        },
        "DQN": {
            "Aggressive": "workspace/models/exp1/F_Low_freq_aggr_DQN_s200.zip",
            "Conservative": "workspace/models/exp1/H_Low_freq_cons_DQN_s200.zip"
        },
    }
}

# --- PHYSICS & REWARDS CONFIGS ---
PHYSICS_CONFIGS = {
    "High_Frequency": { "simulation_frequency": 15, "policy_frequency": 5, "duration": 60, "vehicles_density": 1.5, "lanes_count": 4 },
    "Low_Frequency":  { "simulation_frequency": 15, "policy_frequency": 1, "duration": 60, "vehicles_density": 1.5, "lanes_count": 4 }
}

REWARD_CONFIGS = {
    "High_Frequency": {
        "Aggressive":   { "collision_reward": -30, "right_lane_reward": 0, "high_speed_reward": 0.6, "lane_change_reward": 0, "reward_speed_range": [15, 35], "normalize_reward": False },
        "Conservative": { "collision_reward": -40, "right_lane_reward": 0, "high_speed_reward": 0.3, "lane_change_reward": -0.1, "reward_speed_range": [10, 30], "normalize_reward": False }
    },
    "Low_Frequency": {
        "Aggressive":   { "collision_reward": -15, "right_lane_reward": 0, "high_speed_reward": 0.5, "lane_change_reward": 0.1, "reward_speed_range": [25, 35], "normalize_reward": False },
        "Conservative": { "collision_reward": -20, "right_lane_reward": 0, "high_speed_reward": 0.4, "lane_change_reward": 0, "reward_speed_range": [20, 30], "normalize_reward": False }
    }
}

def make_env(freq_type, behavior_type, rank, seed=0):
    def _init():
        import gymnasium as gym
        import highway_env
        
        # Base Config con corrección de features (7 features)
        base_config = {
            "observation": {
                "type": "Kinematics",
                "vehicles_count": 15,
                "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
                "features_range": {"x": [-100, 100], "y": [-100, 100], "vx": [-20, 20], "vy": [-20, 20]},
                "absolute": False
            }
        }
        
        config = {**base_config, **PHYSICS_CONFIGS[freq_type], **REWARD_CONFIGS[freq_type][behavior_type]}
        
        env = gym.make("highway-v0", render_mode=None, config=config)
        env.reset(seed=seed + rank)
        return env
    return _init

def evaluate_model_parallel(freq_type, algo_type, behavior, path):
    if not os.path.exists(path):
        print(f"⚠️ Warning: Model not found at {path}. Skipping.")
        return []

    model_id = f"{freq_type}_{algo_type}_{behavior}"
    
    # Creamos entorno paralelo
    env = SubprocVecEnv([make_env(freq_type, behavior, i) for i in range(N_CPU)])
    
    try:
        if algo_type == "QRDQN":
            model = QRDQN.load(path, env=env)
        elif algo_type == "DQN":
            model = DQN.load(path, env=env)
        else:
            return []
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return []

    # Evaluation Loop
    episode_results = []
    current_rewards = np.zeros(N_CPU)
    current_speeds = [[] for _ in range(N_CPU)]
    current_crashes = np.zeros(N_CPU, dtype=bool)
    
    obs = env.reset()
    episodes_completed = 0
    
    # --- 2. BARRA DE CARGA TQDM ---
    # Creamos la barra para ESTE modelo específico
    pbar = tqdm(total=N_EPISODES, desc=f"Testing {model_id}", unit="ep")
    
    while episodes_completed < N_EPISODES:
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)
        
        for i in range(N_CPU):
            current_rewards[i] += rewards[i]
            
            speed = infos[i].get('speed', 0)
            crashed = infos[i].get('crashed', False)
            
            current_speeds[i].append(speed)
            if crashed:
                current_crashes[i] = True
                
            if dones[i]:
                # Si aún necesitamos episodios para llegar a N_EPISODES
                if episodes_completed < N_EPISODES:
                    avg_speed = np.mean(current_speeds[i]) if current_speeds[i] else 0
                    
                    episode_results.append({
                        "Frequency": freq_type,
                        "Algorithm": algo_type,
                        "Behavior": behavior,
                        "Model_ID": model_id,
                        "Reward": current_rewards[i],
                        "Avg_Speed": avg_speed,
                        "Crashed": 1 if current_crashes[i] else 0
                    })
                    
                    # Actualizamos contadores y barra
                    episodes_completed += 1
                    pbar.update(1)
                
                # Reseteamos buffers de este núcleo
                current_rewards[i] = 0
                current_speeds[i] = []
                current_crashes[i] = False

    pbar.close() # Cerramos la barra al terminar el modelo
    env.close()
    return episode_results

if __name__ == "__main__":
    all_episodes_data = []

    print(f"--- INICIANDO EVALUACIÓN (N={N_EPISODES}) ---")
    # Iterate through all 8 models
    for freq_key, algo_dict in MODELS.items():
        for algo_key, behavior_dict in algo_dict.items():
            for behavior_key, model_path in behavior_dict.items():
                
                results = evaluate_model_parallel(freq_key, algo_key, behavior_key, model_path)
                all_episodes_data.extend(results)

    if not all_episodes_data:
        print("No results generated.")
        exit()

    df = pd.DataFrame(all_episodes_data)

    # --- METRICS & PLOTTING ---
    summary = df.groupby(['Frequency', 'Algorithm', 'Behavior']).agg(
        Total_Episodes=('Crashed', 'count'),
        Crash_Rate=('Crashed', 'mean'),
        Avg_Reward=('Reward', 'mean'),
        Std_Reward=('Reward', 'std'),
        Avg_Speed=('Avg_Speed', 'mean')
    ).reset_index()

    print("\n--- FINAL METRICS ---")
    print(summary)
    
    summary.to_csv(f"{RESULTS_DIR}/final_metrics_summary.csv", index=False)
    df.to_csv(f"{RESULTS_DIR}/raw_simulation_data.csv", index=False)

    sns.set_theme(style="whitegrid")

    # A. CRASHES
    plt.figure(figsize=(12, 7))
    for label, grp in df.groupby("Model_ID"):
        grp_sorted = grp.reset_index(drop=True)
        plt.plot(np.cumsum(grp_sorted['Crashed']), label=label, linewidth=2)
    plt.title(f"Accumulated Crashes (N={N_EPISODES})")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/plot_accumulated_crashes.png")
    plt.close()

    # B. SPEED BOXPLOT
    df['Hue_Group'] = df['Algorithm'] + " - " + df['Behavior']
    plt.figure(figsize=(12, 7))
    sns.boxplot(data=df, x="Frequency", y="Avg_Speed", hue="Hue_Group", palette="viridis")
    plt.title(f"Average Speed Distribution (N={N_EPISODES})")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/plot_speed_boxplot.png")
    plt.close()

    # C. SPEED HISTOGRAM
    g = sns.FacetGrid(df, col="Frequency", row="Algorithm", hue="Behavior", height=4, aspect=1.5)
    g.map(sns.kdeplot, "Avg_Speed", fill=True, alpha=0.3)
    g.add_legend()
    plt.savefig(f"{RESULTS_DIR}/plot_speed_histogram.png")
    plt.close()

    print(f"\n✅ Done. Check {RESULTS_DIR}")