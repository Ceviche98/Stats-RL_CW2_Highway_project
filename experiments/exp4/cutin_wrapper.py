import gymnasium as gym
import numpy as np

# highway-env imports
from highway_env.vehicle.behavior import IDMVehicle


class BlindSpotCutInWrapper(gym.Wrapper):
    """
    Experiment 4: Scripted adversarial NPC performs an aggressive cut-in when ego approaches to overtake.

    Adds to info dict each step:
      - info["exp4/cutin_started"] : bool
      - info["exp4/ttc"]          : float (seconds, np.inf if not applicable)
      - info["exp4/min_ttc"]      : float (min TTC so far in episode)
      - info["exp4/reaction_time_steps"] : int or None (first brake action after cut-in)
    """

    def __init__(
        self,
        env,
        cutin_trigger_distance_m: float = 5.0,   # "< 5m" from proposal
        spawn_ahead_m: float = 18.0,
        npc_speed_delta: float = -1.0,          # npc slightly slower => encourages overtake
        force_same_side: str | None = None,     # "left" / "right" or None
        brake_action_index: int = 4,            # DiscreteMetaAction: 4 is typically "SLOWER"
        verbose: bool = False
    ):
        super().__init__(env)
        self.cutin_trigger_distance_m = float(cutin_trigger_distance_m)
        self.spawn_ahead_m = float(spawn_ahead_m)
        self.npc_speed_delta = float(npc_speed_delta)
        self.force_same_side = force_same_side
        self.brake_action_index = int(brake_action_index)
        self.verbose = verbose

        # Episode state
        self.adversary = None
        self.cutin_started = False
        self.cutin_step = None
        self.reaction_time_steps = None
        self.min_ttc = np.inf
        self._step_count = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._step_count = 0
        self.cutin_started = False
        self.cutin_step = None
        self.reaction_time_steps = None
        self.min_ttc = np.inf

        self._spawn_adversary()

        # Provide initial info fields
        info = dict(info)
        info.update(self._make_info(ttc=np.inf))
        return obs, info

    def step(self, action):
        self._step_count += 1

        # Before stepping, potentially trigger cut-in based on current positions
        self._maybe_trigger_cutin()

        obs, reward, terminated, truncated, info = self.env.step(action)

        # After stepping, compute TTC and update metrics
        ttc = self._compute_ttc_seconds()

        if ttc < self.min_ttc:
            self.min_ttc = ttc

        # Reaction time: first time agent brakes after cut-in begins
        if self.cutin_started and self.reaction_time_steps is None:
            if int(action) == self.brake_action_index:
                self.reaction_time_steps = self._step_count - (self.cutin_step or self._step_count)

        info = dict(info)
        info.update(self._make_info(ttc=ttc))

        return obs, reward, terminated, truncated, info

    # --------------------- internal helpers ---------------------

    def _unwrapped_env(self):
        return self.env.unwrapped

    def _ego(self):
        return self._unwrapped_env().vehicle

    def _road(self):
        return self._unwrapped_env().road

    def _spawn_adversary(self):
        """
        Spawn a single IDMVehicle in adjacent lane, slightly ahead of ego.
        """
        env = self._unwrapped_env()
        ego = env.vehicle
        road = env.road

        # Find adjacent lane: prefer left/right depending on config
        ego_lane_index = ego.lane_index  # tuple like (from, to, lane_id) in highway-env
        ego_lane_id = ego_lane_index[-1]

        lanes_count = getattr(env.config, "lanes_count", None)
        if lanes_count is None:
            lanes_count = env.config.get("lanes_count", 4)

        # Decide side
        candidates = []
        if ego_lane_id - 1 >= 0:
            candidates.append(("left", ego_lane_id - 1))
        if ego_lane_id + 1 < lanes_count:
            candidates.append(("right", ego_lane_id + 1))

        if not candidates:
            # No adjacent lanes, can't run experiment properly
            self.adversary = None
            if self.verbose:
                print("[Exp4] No adjacent lane available to spawn adversary.")
            return

        if self.force_same_side in ("left", "right"):
            lane_id = None
            for side, cid in candidates:
                if side == self.force_same_side:
                    lane_id = cid
                    break
            if lane_id is None:
                lane_id = candidates[0][1]
        else:
            lane_id = candidates[0][1]

        # Construct adversary lane index (same road segment, different lane id)
        adv_lane_index = (ego_lane_index[0], ego_lane_index[1], lane_id)

        # Place adversary ahead of ego along x
        # position is (x, y) in world coords
        ego_x, ego_y = float(ego.position[0]), float(ego.position[1])

        lane = road.network.get_lane(adv_lane_index)
        # Project point on lane at ego's longitudinal + spawn_ahead
        adv_longitudinal = float(lane.local_coordinates(ego.position)[0] + self.spawn_ahead_m)
        adv_pos = lane.position(adv_longitudinal, 0)

        # Create adversary vehicle with similar speed
        adv_speed = max(0.0, float(ego.speed + self.npc_speed_delta))
        self.adversary = IDMVehicle(road, adv_pos, heading=ego.heading, speed=adv_speed)
        self.adversary.target_lane_index = adv_lane_index

        # Add to road vehicles list
        road.vehicles.append(self.adversary)

        if self.verbose:
            print(f"[Exp4] Spawned adversary in lane {lane_id} at +{self.spawn_ahead_m:.1f}m.")

    def _maybe_trigger_cutin(self):
        """
        Trigger cut-in when ego is close behind adversary and in different lane.
        """
        if self.adversary is None or self.cutin_started:
            return

        ego = self._ego()
        adv = self.adversary

        # Must be in different lanes to be a "blind-spot" adjacent cut-in
        if ego.lane_index == adv.lane_index:
            return

        # Relative longitudinal distance based on x (good enough for straight highway)
        dx = float(adv.position[0] - ego.position[0])  # >0 means adv ahead
        if dx <= 0:
            return  # ego already ahead, don't cut in

        # Trigger when ego gets very close behind adv
        if dx < self.cutin_trigger_distance_m:
            # Force adversary to cut into ego lane NOW
            try:
                adv.target_lane_index = ego.lane_index
            except Exception:
                pass

            self.cutin_started = True
            self.cutin_step = self._step_count

            if self.verbose:
                print(f"[Exp4] Cut-in triggered at step {self._step_count}, dx={dx:.2f}m.")

    def _compute_ttc_seconds(self) -> float:
        """
        TTC = distance / closing_speed when adversary is in (or moving into) ego lane and ahead.
        Returns np.inf if not closing or not applicable.
        """
        if self.adversary is None:
            return np.inf

        ego = self._ego()
        adv = self.adversary

        # Only meaningful if adv is (trying to be) in ego lane
        in_same_lane_or_targeting = (adv.lane_index == ego.lane_index) or (getattr(adv, "target_lane_index", None) == ego.lane_index)
        if not in_same_lane_or_targeting:
            return np.inf

        dx = float(adv.position[0] - ego.position[0])
        if dx <= 0:
            return np.inf  # already passed / overlapping

        closing_speed = float(ego.speed - adv.speed)
        if closing_speed <= 1e-6:
            return np.inf

        return dx / closing_speed

    def _make_info(self, ttc: float):
        return {
            "exp4/cutin_started": bool(self.cutin_started),
            "exp4/ttc": float(ttc) if np.isfinite(ttc) else float("inf"),
            "exp4/min_ttc": float(self.min_ttc) if np.isfinite(self.min_ttc) else float("inf"),
            "exp4/reaction_time_steps": None if self.reaction_time_steps is None else int(self.reaction_time_steps),
        }

