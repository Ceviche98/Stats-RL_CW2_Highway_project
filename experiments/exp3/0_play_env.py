import gymnasium as gym
import highway_env
import pygame

def play_merge_scenario():
    # 1. SETUP ENV
    env = gym.make("merge-v0", render_mode="human")

    # 2. APPLY THE EXACT EXPERIMENT 3 CONFIGURATION
    # This ensures you face the exact same physics/rewards as the AI.
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
        "vehicles_density": 1.5, # High density
        "duration": 20,          # 20 seconds (Short!)
        "simulation_frequency": 15,
        "policy_frequency": 1,   # You will feel the 1-second delay between actions
        
        # We enable manual control just for this script
        "manual_control": True,
        "real_time_rendering": True
    }
    
    env.unwrapped.configure(env_config)
    env.reset()
    
    # 3. CONTROL MAPPING (DiscreteMetaAction)
    # 0: LANE_LEFT
    # 1: IDLE
    # 2: LANE_RIGHT
    # 3: FASTER
    # 4: SLOWER
    
    done = False
    truncated = False
    
    print("--- MANUAL CONTROL START ---")
    print("Use ARROW KEYS to drive.")
    print("UP: Faster | DOWN: Slower | LEFT: Merge Left | RIGHT: Right")
    print("Goal: Merge onto the highway and reach 20-30 m/s without crashing.")
    
    while not (done or truncated):
        # Default action: Idle
        action = 1 
        
        # Capture Pygame Events for Keyboard Input
        keys = pygame.key.get_pressed()
        
        if keys[pygame.K_UP]:
            action = 3
        elif keys[pygame.K_DOWN]:
            action = 4
        elif keys[pygame.K_LEFT]:
            action = 0
        elif keys[pygame.K_RIGHT]:
            action = 2
            
        # Step the environment
        obs, reward, done, truncated, info = env.step(action)
        
        # Optional: Print speed to check if you are in the reward zone
        speed_kmh = info.get('speed', 0) * 3.6 if 'speed' in info else 0
        # For some versions, info['speed'] might not exist, use env.vehicle.speed
        try:
            speed_ms = env.unwrapped.vehicle.speed
            speed_kmh = speed_ms * 3.6
            print(f"Action: {action} | Speed: {speed_kmh:.1f} km/h | Reward: {reward:.2f}", end="\r")
        except:
            pass

        env.render()

    print("\n--- GAME OVER ---")
    env.close()

if __name__ == "__main__":
    play_merge_scenario()