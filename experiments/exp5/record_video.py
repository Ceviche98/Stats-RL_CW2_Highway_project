import os
import imageio.v2 as imageio
from sb3_contrib import QRDQN
from exp5 import WeavingPlatoonEnv

os.environ.pop("SDL_VIDEODRIVER", None)   # important: don’t force dummy
os.environ["SDL_AUDIODRIVER"] = "dummy"

cfg = {
    "simulation_frequency": 15,
    "policy_frequency": 5,
    "duration": 60,
    "normalize_reward": False,
    "offscreen_rendering": True,

    # camera / rendering tweaks (won't affect model load)
    "scaling": 3,
    "centering_position": [0.4, 0.5],
    "manual_control": False,

    "action": {"type": "DiscreteMetaAction"},
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 3,
        "features": ["presence", "x", "y", "vx", "vy"],
        "absolute": False,
    },
}

env = WeavingPlatoonEnv(cfg, render_mode="rgb_array")


model_path = "models/exp5/exp5_weaving_snake_qrdqn.zip"
assert os.path.exists(model_path), f"Model file not found: {model_path}"
model = QRDQN.load(model_path, env=env)


frames = []
obs, info = env.reset()
terminated = truncated = False

while not (terminated or truncated):
    frame = env.render()  # get RGB frame directly
    if frame is not None:
        frames.append(frame)

    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)

env.close()

imageio.mimsave("weaving_snake_manual.mp4", frames, fps=15)
print("Saved weaving_snake_manual.mp4")
