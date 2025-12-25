import gymnasium as gym
import highway_env
import pprint

def verify_and_play():
    # --- 1. THE "RESEARCH GRADE" CONFIGURATION ---
    # This is the exact config you want to use for training.
    # We test it here first.
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
        
        # --- TRAFFIC SETTINGS (The ones that were broken) ---
        "vehicles_count": 50,      # Max cars in the pool
        "vehicles_density": 2,   # Spacing multiplier (Higher = More Cars visible)
        "duration": 40,            # Longer duration for testing
        
        # --- REWARDS (Aggressive) ---
        "collision_reward": -1,    
        "right_lane_reward": 0.05, 
        "high_speed_reward": 0.6,
        "lane_change_reward": 0.05,
        "reward_speed_range": [20, 30],
        "normalize_reward": True,

        "manual_control": True,
        # --- PHYSICS ---
        "simulation_frequency": 15,
        "policy_frequency": 1,
        
        # --- VISUALIZATION ---
        "render_mode": "rgb_array",    # Opens a Window
        "screen_width": 800,
        "screen_height": 300,
        "centering_position": [0.3, 0.5],
        "scaling": 3.5,
        "show_trajectories": False,
        "real_time_rendering": True # Slows down sim so you can watch
    }

    print("🎮 Launching Environment Visualizer...")
    print(f"Target Config: 50 Cars, Density 2.5")

    env = gym.make("highway-v0", render_mode="rgb_array")
    
    # 2. FORCE CONFIG
    env.unwrapped.configure(env_config)
    env.reset()
    
    # 3. NOW Enable Human Rendering manually
    # We do this by interacting with the viewer directly if needed, 
    # but simply switching the render mode attribute often works better in this specific env.
    try:
        env.unwrapped.render_mode = "human"
    except AttributeError:
        pass # Some versions handle this differently
    
    # 4. Loop
    done = False
    truncated = False
    t = 0
    
    print("\n--- CONTROLS ---")
    print("Use ARROW KEYS to drive.")
    print("Press 'S' to save a screenshot.")
    print("----------------\n")

    while not (done or truncated):
        # The environment handles keyboard input automatically in 'human' mode
        # If no key is pressed, it just steps forward.
        
        # Render the frame
        env.render()
        
        # Step the environment (Action is handled internally by manual control usually, 
        # but here we pass 'idle' if we want to watch, or query keyboard)
        # Note: highway-env often needs a specific manual control wrapper for full driving,
        # but basic rendering allows us to SEE the traffic.
        
        # Just step with a dummy action (1=Idle/Lane Keep) to let simulation run
        # If you want to drive, highway-env usually requires 'manual-control' env ID,
        # but for verifying DENSITY, we just want to watch.
        action = env.action_space.sample() # Random actions to see chaos
        # OR use action = 1 to just drive straight and watch traffic
        
        obs, reward, done, truncated, info = env.step(action)
        t += 1

    print("Simulation finished.")
    env.close()

if __name__ == "__main__":
    verify_and_play()