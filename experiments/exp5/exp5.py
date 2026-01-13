# ============================================================
# EXPERIMENT 5 (FINAL): THE BRAKING SNAKE (2-LANE VERSION)
# Tighter Space + Braking + Weaving
# ============================================================

import os
import numpy as np
import math

from sb3_contrib import QRDQN
from stable_baselines3.common.vec_env import DummyVecEnv

from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.road import Road, RoadNetwork
from highway_env.road.lane import StraightLane, LineType
from highway_env.vehicle.behavior import IDMVehicle

# Ensure no window pops up during training
os.environ["SDL_VIDEODRIVER"] = "dummy"

# ============================================================
# VEHICLE DEFINITIONS
# ============================================================

class WeavingBrakingLeader(IDMVehicle):
    """
    Leader that weaves between 2 lanes (0 and 1) AND brakes/accelerates.
    """
    def __init__(self, road, position, speed=25):
        super().__init__(road, position, 0, speed)
        self.time_counter = 0.0

    def act(self, dt=1 / 15):
        self.time_counter += dt
        
        # --- 1. LATERAL CONTROL (2-Lane Weave) ---
        # Wave oscillates around 0.5 with amplitude > 0.5 to ensure switching
        # Values will swing roughly between -0.1 and 1.1
        wave = 0.5 + 0.6 * math.sin(0.6 * self.time_counter)
        target_lane_index = int(round(wave))
        self.target_lane_index = ("a", "b", np.clip(target_lane_index, 0, 1))

        # --- 2. LONGITUDINAL CONTROL (The Brake) ---
        # Speed oscillates between 18 m/s and 28 m/s
        speed_var = math.sin(0.4 * self.time_counter) 
        self.target_speed = 23 + (5 * speed_var) 
        
        super().act(dt)

class ClumsyFollower(IDMVehicle):
    """
    The Tailgater.
    Requires the agent to be incredibly smooth with braking.
    """
    def __init__(self, road, position, speed=25):
        super().__init__(road, position, 0, speed)
        self.time_headway = 0.6   # Very close following
        self.LANE_CHANGE_DELAY = 1.0 

# ============================================================
# ENVIRONMENT
# ============================================================

class WeavingPlatoonEnv(AbstractEnv):
    """
    2-Lane Platoon with Speed & Steering Challenges.
    """

    def _reset(self):
        self._create_road()
        self._create_vehicles()

    def _create_road(self):
        net = RoadNetwork()
        # CHANGED: Loop only 2 times for 2 lanes
        for y in range(2):
            # Lane 0: Continuous Left, Striped Right
            # Lane 1: Striped Left, Continuous Right
            line_type = [LineType.CONTINUOUS, LineType.STRIPED]
            if y == 1: line_type = [LineType.STRIPED, LineType.CONTINUOUS]
            
            net.add_lane("a", "b", StraightLane(
                [0, y * 4], [1000, y * 4], line_types=line_type, width=4
            ))
        self.road = Road(net)

    def _create_vehicles(self):
        # Start in Lane 0
        lane = self.road.network.get_lane(("a", "b", 0))

        # 1. LEADER (Braking Snake)
        leader = WeavingBrakingLeader(self.road, lane.position(90, 0), speed=25)
        
        # 2. AGENT (Buffer)
        self.vehicle = self.action_type.vehicle_class(
            self.road, lane.position(70, 0), speed=25
        )
        
        # 3. FOLLOWER (Tailgater)
        follower = ClumsyFollower(self.road, lane.position(55, 0), speed=25)

        self.road.vehicles = [leader, self.vehicle, follower]

    def _reward(self, action):
        reward = 0.0
        leader, agent, follower = self.road.vehicles
        
        dist_to_leader = leader.position[0] - agent.position[0]
        dist_to_agent = agent.position[0] - follower.position[0]

        # --- 1. LATERAL ALIGNMENT ---
        # Lanes are now only 0 or 1
        leader_lane = round(leader.position[1] / 4)
        agent_lane = round(agent.position[1] / 4)
        
        if leader_lane == agent_lane:
            reward += 0.5
        else:
            reward -= 0.5

        # --- 2. LONGITUDINAL ALIGNMENT ---
        if 15 < dist_to_leader < 25:
            reward += 1.0
        elif dist_to_leader < 10:
            reward -= 1.0 # Penalize getting too close during braking

        # --- 3. FOLLOWER PROTECTION ---
        if 10 < dist_to_agent < 20:
            reward += 1.0
        
        # --- 4. PENALTIES ---
        if dist_to_agent > 30: reward -= 0.5
        if action == 0 or action == 2: reward -= 0.05

        # --- 5. CRASHES ---
        if agent.crashed: reward -= 100
        if follower.crashed: reward -= 150

        return reward

    def _is_terminated(self):
        leader, agent, _ = self.road.vehicles
        dist = leader.position[0] - agent.position[0]
        
        if any(v.crashed for v in self.road.vehicles): return True
        if dist > 50 or dist < -10: return True
            
        return False

    def _is_truncated(self):
        return self.time >= self.config["duration"]

# ============================================================
# TRAINING SETUP
# ============================================================

def make_env():
    return WeavingPlatoonEnv({
        "simulation_frequency": 15,
        "policy_frequency": 5,
        "duration": 40,
        "action": {"type": "DiscreteMetaAction"}, 
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 3, 
            "features": ["presence", "x", "y", "vx", "vy"], 
            "absolute": False 
        }
    })

if __name__ == "__main__":
    env = DummyVecEnv([make_env])

    model = QRDQN(
        "MlpPolicy",
        env,
        learning_rate=1e-4,
        buffer_size=350000,
        learning_starts=500,
        batch_size=128,
        gamma=0.9,
        target_update_interval=50,
        train_freq=1,
        gradient_steps=1,
        exploration_fraction=0.3,
        exploration_final_eps=0.05,
        verbose=1
    )

    print("\n=== Training Exp 5: 2-Lane Braking Snake ===\n")
    model.learn(total_timesteps=300_000) 
    model.save("exp5_weaving_snake_qrdqn")
    print("Done. Verify the agent handles the tight 2-lane squeeze.")