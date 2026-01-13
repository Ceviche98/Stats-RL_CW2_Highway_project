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
from tqdm import tqdm 

# --- 1. SILENCIAR WARNINGS ---
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# --- CONFIGURATION ---
# Range: 1.3 to 2.5 with 0.2 step -> [1.3, 1.5, 1.7, 1.9, 2.1, 2.3, 2.5]
TEST_DENSITIES = [round(x, 1) for x in np.arange(1.3, 2.6, 0.2)]
N_EPISODES = 1000   
N_CPU = 12 
script_dir = os.path.abspath(".")
RESULTS_subdir = "/results/exp3_generalization/"
RESULTS_DIR = os.path.join(script_dir, RESULTS_subdir)
os.makedirs(RESULTS_DIR, exist_ok=True)
MODELS_DIR=os.path.join(script_dir, "models/")

# Updated to use correct model names from TABLE III
# Aggressive Low-Freq: E (DQN), F (QRDQN - BEST BASELINE)
# Conservative Low-Freq: G (DQN), H (QRDQN)
MODELS = {
    "Low_Frequency": {
        "QRDQN": {
            "Aggressive": os.path.join(MODELS_DIR, "exp1/F_Low_freq_aggr_QRDQN_s200.zip"),
            "Conservative": os.path.join(MODELS_DIR, "exp1/H_Low_freq_cons_QRDQN_s200.zip")
        },
        "DQN": {
            "Aggressive": os.path.join(MODELS_DIR, "exp1/E_Low_freq_aggr_DQN_s200.zip"),
            "Conservative": os.path.join(MODELS_DIR, "exp1/G_Low_freq_cons_DQN_s200.zip")
        },
    }
}


# --- ENV BASE CONFIGS ---
ENV_BASE_CONFIGS = {
    "High_Frequency": { "simulation_frequency": 15, "policy_frequency": 5, "duration": 60, "lanes_count": 4 },
    "Low_Frequency":  { "simulation_frequency": 15, "policy_frequency": 1, "duration": 60, "lanes_count": 4 }
}

# --- REWARD CONFIGS (CRUCIAL FIX) ---
# Separamos las recompensas para evaluar el modelo en su "juego" correcto
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

def make_density_env(freq_type, behavior_type, density, rank, seed=0):
    def _init():
        import gymnasium as gym
        import highway_env
        
        # 1. Base Config con 7 features (Evita error de observation space)
        base_config = {
            "observation": {
                "type": "Kinematics",
                "vehicles_count": 15,
                "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
                "features_range": {"x": [-100, 100], "y": [-100, 100], "vx": [-20, 20], "vy": [-20, 20]},
                "absolute": False
            }
        }
        
        # 2. Obtener config de frecuencia base
        freq_conf = ENV_BASE_CONFIGS[freq_type]
        
        # 3. Obtener config de recompensa ESPECIFICA
        reward_conf = REWARD_CONFIGS[freq_type][behavior_type]
        
        # 4. Combinar todo e inyectar la densidad variable
        config = {
            **base_config, 
            **freq_conf, 
            **reward_conf,
            "vehicles_density": density # Sobrescribimos la densidad
        }
        
        env = gym.make("highway-v0", render_mode=None, config=config)
        env.reset(seed=seed + rank)
        return env
    return _init

def evaluate_density_parallel(freq_type, algo_type, behavior, path, density):
    if not os.path.exists(path):
        return []

    model_id = f"{freq_type}_{algo_type}_{behavior}"
    
    # 1. Create Parallel Environments (Pasando behavior para cargar rewards correctos)
    env = SubprocVecEnv([make_density_env(freq_type, behavior, density, i) for i in range(N_CPU)])
    
    # 2. Load Model
    try:
        if algo_type == "QRDQN":
            model = QRDQN.load(path, env=env)
        elif algo_type == "DQN":
            model = DQN.load(path, env=env)
        else:
            return []
    except:
        return []

    # 3. Eval Loop
    episode_results = []
    
    current_rewards = np.zeros(N_CPU)
    current_speeds = [[] for _ in range(N_CPU)]
    current_crashes = np.zeros(N_CPU, dtype=bool)
    episodes_completed = 0
    obs = env.reset()
    
    # Barra de progreso interna (opcional, si quieres ver detalle por densidad)
    # pbar = tqdm(total=N_EPISODES, desc=f"  > {model_id} @ {density}", leave=False)
    
    while episodes_completed < N_EPISODES:
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)
        
        for i in range(N_CPU):
            current_rewards[i] += rewards[i]
            if "speed" in infos[i]:
                current_speeds[i].append(infos[i]['speed'])
                
            if infos[i].get('crashed', False):
                current_crashes[i] = True
                
            if dones[i]:
                if episodes_completed < N_EPISODES:
                    avg_speed_ms = np.mean(current_speeds[i]) if current_speeds[i] else 0
                    
                    episode_results.append({
                        "Model_ID": model_id,
                        "Frequency": freq_type,
                        "Algorithm": algo_type,
                        "Behavior": behavior,
                        "Density": density, 
                        "Reward": current_rewards[i],
                        "Avg_Speed_KMH": avg_speed_ms * 3.6,
                        "Crashed": 1 if current_crashes[i] else 0,
                        "Success": 0 if current_crashes[i] else 1
                    })
                    episodes_completed += 1
                    # pbar.update(1)
                
                current_rewards[i] = 0
                current_speeds[i] = []
                current_crashes[i] = False
    
    # pbar.close()
    env.close()
    return episode_results

if __name__ == "__main__":
    all_results = []
    
    # Calculamos total de tareas
    total_configs = 8 * len(TEST_DENSITIES)
    
    print(f"--- STARTING GENERALIZATION TEST (N={N_EPISODES}) ---")
    print(f"Densities: {TEST_DENSITIES}")
    print(f"Total Configurations: {total_configs}")
    
    # Barra de progreso GLOBAL
    pbar_global = tqdm(total=total_configs, desc="Global Progress", unit="cfg")
    
    for freq_key, algo_dict in MODELS.items():
        for algo_key, behavior_dict in algo_dict.items():
            for behavior_key, model_path in behavior_dict.items():
                
                for density in TEST_DENSITIES:
                    pbar_global.set_description(f"Testing {freq_key[:4]}_{algo_key}_{behavior_key[:4]} @ D={density}")
                    
                    res = evaluate_density_parallel(freq_key, algo_key, behavior_key, model_path, density)
                    all_results.extend(res)
                    pbar_global.update(1)
    
    pbar_global.close()

    if not all_results:
        print("No results generated.")
        exit()

    # --- DATAFRAME ---
    df = pd.DataFrame(all_results)
    df.to_csv(f"{RESULTS_DIR}/exp3_full_raw.csv", index=False)

    # --- METRICS & CONFIDENCE INTERVALS ---
    summary = df.groupby(['Model_ID', 'Density', 'Frequency', 'Algorithm', 'Behavior']).agg(
        N=('Success', 'count'),
        Success_Rate=('Success', 'mean'),
        Avg_Reward=('Reward', 'mean'),
        Avg_Speed_KMH=('Avg_Speed_KMH', 'mean'),
        Std_Speed=('Avg_Speed_KMH', 'std')
    ).reset_index()

    summary['CI_Error'] = 1.96 * np.sqrt(
        (summary['Success_Rate'] * (1 - summary['Success_Rate'])) / summary['N']
    )
    
    summary.to_csv(f"{RESULTS_DIR}/exp3_metrics_summary.csv", index=False)

    # --- PLOTTING ---
    sns.set_theme(style="whitegrid")
    
    # 1. SUCCESS RATE
    plt.figure(figsize=(14, 8))
    sns.lineplot(
        data=summary, 
        x="Density", y="Success_Rate", 
        hue="Model_ID", style="Frequency",
        markers=True, dashes=True, linewidth=2.5, palette="tab10"
    )
    for name, group in summary.groupby("Model_ID"):
        plt.fill_between(
            group["Density"],
            group["Success_Rate"] - group["CI_Error"],
            group["Success_Rate"] + group["CI_Error"],
            alpha=0.1
        )
    plt.title(f"Success Rate vs Density (N={N_EPISODES})")
    plt.ylabel("Success Rate")
    plt.ylim(-0.05, 1.05)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/plot_success_full.png")
    plt.close()

    # 2. SPEED PLOT
    plt.figure(figsize=(14, 8))
    sns.lineplot(
        data=summary, 
        x="Density", y="Avg_Speed_KMH", 
        hue="Model_ID", style="Frequency",
        markers=True, linewidth=2.5, palette="viridis"
    )
    plt.title("Avg Speed vs Density")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/plot_speed_full.png")
    plt.close()

    print(f"\n✅ FINISHED! Results in {RESULTS_DIR}")