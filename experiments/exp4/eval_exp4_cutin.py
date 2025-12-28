import os
import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from stable_baselines3 import DQN
from sb3_contrib import QRDQN

from cutin_wrapper import BlindSpotCutInWrapper


# ---- Paths: adjust if your model names differ ----
MODELS = {
    "DQN": "../../models/exp1_DQN_s42.zip",
    "QRDQN": "../../models/exp1_QRDQN_s42.zip",
}

RESULTS_DIR = "../../results/exp4_cutin"
os.makedirs(RESULTS_DIR, exist_ok=True)


def make_env(seed: int = 0):
    # Use same env config style as exp1 if you want consistency:
    env_config = {
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 15,
            "features": ["presence", "x", "y", "vx", "vy"],
            "absolute": False,
        },
        "action": {"type": "DiscreteMetaAction"},
        "simulation_frequency": 15,
        "policy_frequency": 1,
        "lanes_count": 4,
        "render_mode": None,
    }

    env = gym.make("highway-v0", render_mode=None, config=env_config)
    env.reset(seed=seed)

    # Wrap with Exp4 adversary
    env = BlindSpotCutInWrapper(
        env,
        cutin_trigger_distance_m=5.0,  # proposal constraint
        spawn_ahead_m=18.0,
        npc_speed_delta=-1.0,
        brake_action_index=4,          # "SLOWER" in DiscreteMetaAction
        verbose=False
    )
    return env


def run_eval(model_name: str, model_path: str, episodes: int = 200, seed: int = 0):
    env = make_env(seed=seed)

    if model_name == "DQN":
        model = DQN.load(model_path, env=env)
    else:
        model = QRDQN.load(model_path, env=env)

    rows = []

    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep)
        done = False

        ep_min_ttc = np.inf
        cutin_happened = False
        reaction_steps = None
        crashed = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Pull exp4 signals from info
            cutin_happened = cutin_happened or bool(info.get("exp4/cutin_started", False))
            ttc = info.get("exp4/ttc", float("inf"))
            if np.isfinite(ttc) and ttc < ep_min_ttc:
                ep_min_ttc = float(ttc)

            if reaction_steps is None:
                reaction_steps = info.get("exp4/reaction_time_steps", None)

            if info.get("crashed", False):
                crashed = True

        rows.append({
            "Model": model_name,
            "Episode": ep,
            "CutInHappened": cutin_happened,
            "MinTTC": ep_min_ttc if np.isfinite(ep_min_ttc) else np.nan,
            "ReactionTimeSteps": reaction_steps if reaction_steps is not None else np.nan,
            "Crashed": crashed
        })

    env.close()
    return pd.DataFrame(rows)


def plot_histograms(df: pd.DataFrame):
    # TTC histogram
    plt.figure(figsize=(8, 5))
    for model in df["Model"].unique():
        vals = df[(df["Model"] == model) & (df["CutInHappened"]) واپس if False else df[(df["Model"] == model) & (df["CutInHappened"])]

        vals = vals["MinTTC"].dropna()
        plt.hist(vals, bins=30, alpha=0.5, label=model)
    plt.title("Experiment 4: Min TTC Distribution (Cut-in Episodes)")
    plt.xlabel("Min TTC (seconds)")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(RESULTS_DIR, "ttc_hist.png"), dpi=200)

    # Reaction time histogram
    plt.figure(figsize=(8, 5))
    for model in df["Model"].unique():
        vals = df[(df["Model"] == model) & (df["CutInHappened"])]["ReactionTimeSteps"].dropna()
        plt.hist(vals, bins=30, alpha=0.5, label=model)
    plt.title("Experiment 4: Reaction Time (steps) after Cut-in")
    plt.xlabel("Reaction time (steps) — first brake action")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(RESULTS_DIR, "reaction_time_hist.png"), dpi=200)


def summarize(df: pd.DataFrame):
    # Focus on episodes where cut-in occurred at least once
    cutin_df = df[df["CutInHappened"]]

    summary = cutin_df.groupby("Model").agg(
        Episodes=("Episode", "count"),
        CrashRate=("Crashed", "mean"),
        MeanMinTTC=("MinTTC", "mean"),
        MedianMinTTC=("MinTTC", "median"),
        MeanReactionSteps=("ReactionTimeSteps", "mean"),
        MedianReactionSteps=("ReactionTimeSteps", "median"),
    ).reset_index()

    summary_path = os.path.join(RESULTS_DIR, "exp4_summary.csv")
    summary.to_csv(summary_path, index=False)

    full_path = os.path.join(RESULTS_DIR, "exp4_all_episodes.csv")
    df.to_csv(full_path, index=False)

    print("Saved:", summary_path)
    print("Saved:", full_path)
    print(summary)


if __name__ == "__main__":
    all_dfs = []

    for model_name, model_path in MODELS.items():
        if not os.path.exists(model_path):
            print(f"Missing model file: {model_path} (skipping {model_name})")
            continue

        print(f"Evaluating {model_name} from {model_path}")
        df = run_eval(model_name, model_path, episodes=200, seed=123)
        all_dfs.append(df)

    if not all_dfs:
        raise RuntimeError("No models evaluated. Check MODELS paths.")

    df_all = pd.concat(all_dfs, ignore_index=True)

    summarize(df_all)
    plot_histograms(df_all)

    print("Plots saved to:", RESULTS_DIR)
