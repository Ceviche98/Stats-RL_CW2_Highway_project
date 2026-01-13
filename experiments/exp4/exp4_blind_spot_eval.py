import gymnasium as gym
import numpy as np
import os
import sys
import highway_env
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from gymnasium.wrappers import RecordVideo

# --- CONFIGURATION ---
USE_SAFETY_SHIELD = True  # <--- ENABLED: Forces avoidance
DQN_PATH = "../../models/exp1/E_Low_freq_aggr_DQN_s200.zip"
QRDQN_PATH = "../../models/exp1/F_Low_freq_aggr_QRDQN_s200.zip"
VIDEO_DIR = "../../videos/exp4_blind_spot_shielded/"

os.makedirs(VIDEO_DIR, exist_ok=True)

def get_ttc(ego_vehicle, other_vehicle):
    """Calculate Time-To-Collision."""
    dx = other_vehicle.position[0] - ego_vehicle.position[0]
    dv = ego_vehicle.speed - other_vehicle.speed
    if dv <= 0: return 100.0
    ttc = dx / dv
    return ttc if ttc > 0 else 100.0

def run_visual_test(agent, agent_name, num_episodes=3):
    try:
        env = gym.make("highway-v0", render_mode="rgb_array")
    except gym.error.NameNotFound:
        print("ERROR: highway-v0 not found. Please run: pip install highway-env")
        return

    env.unwrapped.configure({
        "lanes_count": 3,
        "vehicles_density": 1.2,
        "duration": 40,
        "screen_width": 600,
        "screen_height": 300,
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 15,
            "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
            "features_range": {"x": [-100, 100], "y": [-100, 100], "vx": [-20, 20], "vy": [-20, 20]},
            "absolute": False,
            "order": "sorted"
        }
    })

    env = RecordVideo(
        env,
        video_folder=VIDEO_DIR,
        name_prefix=f"{agent_name}_shield_TRUE",
        episode_trigger=lambda e: True
    )

    print(f"\n--- Recording Shielded Video for {agent_name} ---")

    # --- EPISODE-LEVEL LOGS ---
    ep_returns = []
    ep_lengths = []
    ep_shield_uses = []
    ep_crashed = []

    # (Optional) store step-series only for the first episode to keep plots tidy
    first_ep_ttc = None
    first_ep_speed = None
    first_ep_action = None
    first_ep_shield = None

    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        truncated = False

        adversary_triggered = False
        adversary_vehicle = None

        # --- per-episode counters ---
        total_reward = 0.0
        steps = 0
        shield_count = 0
        crashed_flag = False

        # --- step-series (only keep for episode 0) ---
        ttc_series = []
        speed_series = []
        action_series = []
        shield_series = []

        while not (done or truncated):
            # A. PREDICT ACTION
            if agent is not None:
                action, _ = agent.predict(obs, deterministic=True)
                # SB3 often returns numpy arrays; make sure it's an int for logging
                # Robust conversion for SB3 actions: int, 0-D np array, 1-D np array, list
            if isinstance(action, np.ndarray):
                action = int(action.item())  # works for 0-D and 1-element arrays
            elif isinstance(action, (list, tuple)):
                action = int(action[0])
            else:
                action = int(action)

            # B. ADVERSARIAL LOGIC
            ego_vehicle = env.unwrapped.vehicle

            if adversary_vehicle is None:
                for v in env.unwrapped.road.vehicles:
                    if v is ego_vehicle:
                        continue
                    lat_dist = abs(v.position[1] - ego_vehicle.position[1])
                    long_dist = v.position[0] - ego_vehicle.position[0]
                    if 0 < long_dist < 15 and 3 < lat_dist < 5:
                        adversary_vehicle = v
                        break

            if adversary_vehicle is not None and not adversary_triggered:
                current_lane = adversary_vehicle.target_lane_index

                if adversary_vehicle.position[1] > ego_vehicle.position[1]:
                    lane_change = -1
                else:
                    lane_change = 1

                if isinstance(current_lane, tuple):
                    lane_from, lane_to, lane_id = current_lane
                    new_lane_id = max(0, min(2, lane_id + lane_change))
                    adversary_vehicle.target_lane_index = (lane_from, lane_to, new_lane_id)
                else:
                    new_lane = current_lane + lane_change
                    adversary_vehicle.target_lane_index = max(0, min(2, new_lane))

                adversary_triggered = True
                print(f"  > Ep {episode}: Cut-in triggered!")

            # C. SAFETY SHIELD
            shielded = 0
            ttc = 100.0
            if adversary_triggered and USE_SAFETY_SHIELD and adversary_vehicle is not None:
                ttc = get_ttc(ego_vehicle, adversary_vehicle)
                if ttc < 1.5:
                    action = 4  # Emergency brake
                    shielded = 1
                    shield_count += 1

            obs, reward, done, truncated, info = env.step(action)

            # Update episode stats
            total_reward += float(reward)
            steps += 1

            # Crash detection (highway-env typically exposes vehicle.crashed)
            crashed_flag = crashed_flag or bool(getattr(env.unwrapped.vehicle, "crashed", False))

            # Step-series logs (optional)
            ttc_series.append(float(ttc))
            speed_series.append(float(getattr(env.unwrapped.vehicle, "speed", 0.0)))
            action_series.append(int(action))
            shield_series.append(int(shielded))

        # End of episode logs
        ep_returns.append(total_reward)
        ep_lengths.append(steps)
        ep_shield_uses.append(shield_count)
        ep_crashed.append(int(crashed_flag))

        if episode == 0:
            first_ep_ttc = ttc_series
            first_ep_speed = speed_series
            first_ep_action = action_series
            first_ep_shield = shield_series

    env.close()
    print(f"Shielded videos saved to {VIDEO_DIR}")

    # --- PLOTTING ---
    # 1) Episode-level summary plot
    fig = plt.figure()
    x = np.arange(len(ep_returns))
    plt.plot(x, ep_returns, marker="o")
    plt.title(f"{agent_name}: Episode Return")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.grid(True)
    out_path = os.path.join(VIDEO_DIR, f"{agent_name}_episode_return.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure()
    plt.plot(x, ep_shield_uses, marker="o")
    plt.title(f"{agent_name}: Shield Activations per Episode")
    plt.xlabel("Episode")
    plt.ylabel("# Shield overrides")
    plt.grid(True)
    out_path = os.path.join(VIDEO_DIR, f"{agent_name}_shield_uses.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 2) Step-level plot for first episode (TTC + speed + shield)
    if first_ep_ttc is not None:
        t = np.arange(len(first_ep_ttc))

        fig = plt.figure()
        plt.plot(t, first_ep_ttc)
        plt.title(f"{agent_name}: TTC over Time (Episode 0)")
        plt.xlabel("Step")
        plt.ylabel("TTC (s)")
        plt.grid(True)
        out_path = os.path.join(VIDEO_DIR, f"{agent_name}_ep0_ttc.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        fig = plt.figure()
        plt.plot(t, first_ep_speed)
        plt.title(f"{agent_name}: Ego Speed over Time (Episode 0)")
        plt.xlabel("Step")
        plt.ylabel("Speed")
        plt.grid(True)
        out_path = os.path.join(VIDEO_DIR, f"{agent_name}_ep0_speed.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        fig = plt.figure()
        plt.plot(t, first_ep_shield)
        plt.title(f"{agent_name}: Shield Active (Episode 0)")
        plt.xlabel("Step")
        plt.ylabel("Shield (0/1)")
        plt.ylim(-0.1, 1.1)
        plt.grid(True)
        out_path = os.path.join(VIDEO_DIR, f"{agent_name}_ep0_shield.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"Graphs saved to {VIDEO_DIR}")


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    print(">>> RUNNING SHIELDED TEST (Safety Override ENABLED) <<<")
    
    # 1. TEST DQN
    if not os.path.exists(DQN_PATH):
        print(f"ERROR: Could not find DQN model at: {os.path.abspath(DQN_PATH)}")
    else:
        print(f"Loading DQN from {DQN_PATH}...")
        dqn_model = DQN.load(DQN_PATH)
        run_visual_test(dqn_model, "DQN", num_episodes=3)

    # 2. TEST QR-DQN
    if not os.path.exists(QRDQN_PATH):
        print(f"Warning: Could not find QR-DQN model at: {QRDQN_PATH}")
    else:
        print(f"Loading QR-DQN from {QRDQN_PATH}...")
        try:
            from sb3_contrib import QRDQN
            qrdqn_model = QRDQN.load(QRDQN_PATH)
        except:
            print("  (Loading as standard DQN class...)")
            qrdqn_model = DQN.load(QRDQN_PATH)
            
        run_visual_test(qrdqn_model, "QRDQN", num_episodes=3)
