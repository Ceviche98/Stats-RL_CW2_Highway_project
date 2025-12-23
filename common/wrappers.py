import gymnasium as gym
import numpy as np
from collections import deque

class RobustnessWrapper(gym.Wrapper):
    """
    A single wrapper to inject 3 types of degradation for Experiment 2.
    Configured via the 'mode' parameter.
    """
    def __init__(self, env, mode='clean', sigma=0.5, delay=1, blackout_period=5):
        super().__init__(env)
        self.mode = mode
        
        # Parameters
        self.noise_sigma = sigma          # For 'noise' mode
        self.delay = delay                # For 'lag' mode
        self.blackout_period = blackout_period # For 'blackout' mode
        
        # State buffers
        self.action_buffer = deque(maxlen=delay + 1)
        self.step_counter = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.step_counter = 0
        
        # Reset Action Buffer (fill with idle actions initially)
        self.action_buffer.clear()
        for _ in range(self.delay + 1):
            self.action_buffer.append(0) # Assuming 0 is IDLE/LANE_KEEP
            
        return self._process_obs(obs), info

    def step(self, action):
        self.step_counter += 1
        
        # MODE: LAG (Action Delay)
        if self.mode == 'lag':
            self.action_buffer.append(action)
            executed_action = self.action_buffer.popleft()
        else:
            executed_action = action

        # Step the environment
        obs, reward, terminated, truncated, info = self.env.step(executed_action)
        
        # Process Observation (Noise/Blackout)
        processed_obs = self._process_obs(obs)
        
        return processed_obs, reward, terminated, truncated, info

    def _process_obs(self, obs):
        if self.mode == 'clean' or self.mode == 'lag':
            return obs
            
        # Kinematics: [x, y, vx, vy] usually. 
        # highway-env observations are often (N, 5) or (N, 4) arrays.
        
        # MODE: NOISE (Sensor Jitter)
        if self.mode == 'noise':
            noise = np.random.normal(0, self.noise_sigma, size=obs.shape)
            # Only add noise to coordinates (usually indices 1, 2), not ID/Lane
            # But adding to all features is a valid "Sensor Fail" test too.
            return obs + noise

        # MODE: BLACKOUT (Sensor Blinking)
        if self.mode == 'blackout':
            if self.step_counter % self.blackout_period == 0:
                return np.zeros_like(obs) # Blindness
            return obs
            
        return obs