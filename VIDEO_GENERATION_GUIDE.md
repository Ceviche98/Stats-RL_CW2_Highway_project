# Video Generation Guide

## Overview
Two scripts generate MP4 videos showing trained agents driving in highway-v0 with real-time speedometer overlay:
1. **Low-frequency models** (1 Hz): `record_video_s200_speed.py`
2. **High-frequency models** (5 Hz): `record_video_highf_s200_speed.py`

Both scripts produce **60-second episodes** with color-coded speedometer showing reward zone achievement.

---

## Script 1: Low-Frequency Video (`record_video_s200_speed.py`)

### What It Does
- Loads trained low-frequency model (1 decision/second)
- Runs one 60-second episode
- Records RGB frames with speedometer overlay
- Saves MP4 at 15 FPS (~4 seconds of real time per video second)

### How to Run
```bash
cd experiments/exp1
python record_video_s200_speed.py
```

### Output
```
✓ Loading model from models/exp1/F_Low_freq_aggr_QRDQN_s200.zip...
✓ Recording frame by frame...
✓ Saving video with 900 frames...
✓ Video saved to: results/videos/speedometer_F_Low_freq_aggr_QRDQN_s200.mp4
```

### Video Features

**Speedometer Display (Top-Left):**
```
Speed: 28.3 km/h              ← Current agent speed
Target: 90-126 km/h           ← Reward zone (25-35 m/s converted to km/h)
```

**Color Coding:**
- 🟢 **Green**: Speed in reward zone (earning positive reward)
- 🔴 **Red**: Speed too high (above reward zone, +0.5 reward penalty)
- 🔵 **Blue**: Speed too low (below reward zone, -0.3 reward loss)

### Customization Options

#### Try Different Models
```python
# Line 13-15, change model_name:
model_name = "E_Low_freq_aggr_DQN_s200"       # DQN vs QRDQN
model_name = "G_Low_freq_cons_DQN_s200"       # Conservative mode
model_name = "H_Low_freq_cons_QRDQN_s200"     # Conservative QRDQN
model_name = "K_Low_freq_aggressive_QRDQN_s300_opt"  # Optimized variant
```

#### Change Environment Parameters
```python
# Lines 18-47, modify env_config dictionary:

env_config = {
    ...
    # Traffic density (default: 1.5)
    "vehicles_density": 2.0,           # Try: 1.0, 1.5, 2.0, 2.5
    
    # Decision frequency (default: 1 Hz)
    "policy_frequency": 2,             # Try: 1, 2, 5
    
    # Collision penalty (default: -15)
    "collision_reward": -20,           # Try: -10, -20, -30
    
    # Speed reward (default: 0.5)
    "high_speed_reward": 0.7,          # Try: 0.3, 0.5, 0.7
    
    # Target speed range for reward (default: [25, 35] m/s)
    "reward_speed_range": [30, 40],    # Try: [20, 30], [25, 40]
    ...
}
```

#### Change Video Output
```python
# Line 14, modify VIDEO_OUTPUT:
VIDEO_OUTPUT = os.path.join(
    script_dir, 
    "results/videos/my_custom_name.mp4"
)

# Or modify FPS for playback speed:
# Line 122: imageio.mimsave(VIDEO_OUTPUT, frames, fps=30)  # Faster playback
```

### Technical Details

**Frame Capture:**
- Environment rendered at 15 Hz (simulation frequency)
- Policy decides at 1 Hz (1 action per second)
- Video saved at 15 FPS (real-time playback)
- Total frames: ~900 (60 seconds × 15 FPS)

**Speedometer Calculation:**
```python
speed_ms = env.unwrapped.vehicle.speed  # Current velocity in m/s
speed_kmh = speed_ms * 3.6              # Convert to km/h
```

**Color Logic:**
```python
if speed_kmh > target_max:              # Too fast
    color = (0, 0, 255)  # Red
elif speed_kmh >= target_min:           # In reward zone
    color = (0, 255, 0)  # Green
else:                                    # Too slow
    color = (255, 0, 0)  # Blue
```

### Troubleshooting

| Error | Solution |
|---|---|
| `FileNotFoundError: Model not found` | Verify model exists in `models/exp1/` |
| `cv2 error: Layout of output array incompatible` | Already fixed! Uses `np.ascontiguousarray()` |
| `imageio-ffmpeg not found` | Install: `pip install imageio-ffmpeg` |
| `Video won't open` | Use VLC or FFplay; or increase FPS |
| `Segmentation fault` | Check GPU memory; set `device="cpu"` in model |

---

## Script 2: High-Frequency Video (`record_video_highf_s200_speed.py`)

### What It Does
- Loads trained high-frequency model (5 decisions/second)
- Runs 60-second episode with rapid decision updates
- Same speedometer overlay as low-freq version
- Saves MP4 at 5 FPS (demonstrates decision-making frequency)

### How to Run
```bash
cd experiments/exp1
python record_video_highf_s200_speed.py
```

### Output
```
--- Loading Model: models/exp1/I_High_freq_aggressive_QRDQN_s500_opt.zip ---
--- Starting Simulation (Max 60s at 5 Hz) ---
Time: 0.0s | Speed: 15.2 km/h | Reward: -0.05
Time: 1.0s | Speed: 18.5 km/h | Reward: 0.12
...
--- Saving Video to results/videos/speedometer_highfreq_I_High_freq_aggressive_QRDQN_s500_opt.mp4 ---
✓ Done! Video saved to ...
Total reward: -28.3
```

### Key Differences from Low-Freq

| Aspect | Low-Freq (1 Hz) | High-Freq (5 Hz) |
|---|---|---|
| Decision rate | 1 per second | 5 per second |
| Decision interval | 1.0 second | 0.2 seconds |
| Video FPS | 15 FPS | 5 FPS |
| Planning horizon | Longer (predicts 1s ahead) | Shorter (predicts 0.2s ahead) |
| Reaction time | Slower but more deliberate | Faster but more reactive |

### Video Features (High-Freq)

**Observation Range:**
```python
"features_range": {"x": [-100, 100], "y": [-100, 100], ...}
```
Same as low-freq for compatibility.

**Reward Zone (High-Freq Model I):**
```
Speed: 32.1 km/h
Target: 54-126 km/h  (15-35 m/s)
```

### Customization for High-Freq

#### Try Different High-Freq Models
```python
# Line 13, change model_name:
model_name = "I_High_freq_aggressive_QRDQN_s500_opt"   # Aggressive (best)
model_name = "J_High_freq_conservative_QRDQN_s500_opt"  # Conservative
```

#### Modify Environment for High-Freq
```python
# Lines 20-65, in make_eval_env function:

if mode == "aggressive":
    config.update({
        "collision_reward": -30,       # Try: -20, -40
        "high_speed_reward": 0.8,      # Try: 0.5, 1.0
        "reward_speed_range": [15, 35] # Try: [10, 30], [20, 40]
    })
```

#### Change Playback Speed
```python
# Line 126, modify fps:
imageio.mimsave(VIDEO_OUTPUT, frames, fps=10)  # Slower playback
imageio.mimsave(VIDEO_OUTPUT, frames, fps=2)   # Very slow (see individual decisions)
```

### Why 5 FPS for High-Freq Video?

High-freq models make **5 decisions per second**. We capture frames at each decision (5 FPS) to show:
- How often the agent makes decisions
- What the agent "sees" at each decision point
- Rapid course corrections visible in real-time

If we saved at 15 FPS, you'd see every physics frame but not necessarily the agent's decision timing.

---

## Comparison: When to Use Each Script

### Use `record_video_s200_speed.py` When:
- ✅ Demonstrating best model F performance
- ✅ Showing smooth, deliberate driving at 1 Hz
- ✅ Comparing decision quality (not frequency)
- ✅ Analyzing specific driving behaviors
- ✅ Running quick demos (~5 minutes)

**Try these models:**
- F (BEST - 13.8% crash rate)
- K, L, M (Optimized variants)
- E, G, H (Baseline comparisons)

### Use `record_video_highf_s200_speed.py` When:
- ✅ Showing high-frequency decision-making
- ✅ Demonstrating rapid lane changes
- ✅ Comparing frequency impact on performance
- ✅ Analyzing 5 Hz control behavior
- ✅ Highlighting paradox: I (30.4%) worse than F (13.8%)

**Try these models:**
- I (Aggressive, but paradoxically worse)
- J (Conservative, better than baseline)

---

## Complete Workflow

### Generate Multiple Videos for Comparison

```bash
cd experiments/exp1

# Low-frequency baseline comparison
python record_video_s200_speed.py  # Best model (F)

# Then manually change model_name in script to:
# E (DQN), G (Conservative), K (Optimized)
# and run again for each

# High-frequency models
python record_video_highf_s200_speed.py  # Model I
# Change model_name to J and run again
```

### What the Videos Prove

1. **Low-Freq Best Model (F)**: Smooth, high reward achievement
2. **High-Freq Paradox (I vs F)**: Despite 5x faster decisions, worse performance
3. **Conservative Models (G, H)**: Either too timid or overly cautious
4. **Optimized Variants (K, L, M)**: Better balance through hyperparameter tuning

### Create Presentation Compilation

```bash
# After generating videos, you could:
# 1. Keep best few videos
# 2. Use video editor to combine with before/after metrics
# 3. Add audio commentary explaining findings
# 4. Generate GIF previews for paper/report
```

---

## Technical Specifications

### Video Format
- **Container**: MP4 (MPEG-4)
- **Codec**: H.264 (H264)
- **Resolution**: 600×300 pixels (highway-env default)
- **Frame rate**: 15 FPS (low-freq) or 5 FPS (high-freq)
- **Duration**: 60 seconds
- **File size**: ~5-10 MB

### Performance Metrics Displayed
- **Speed**: Real-time velocity in km/h
- **Target zone**: Reward-earning speed range
- **Color feedback**: Instant visual of reward signal

### Environment Constants
- **Episode duration**: 60 seconds
- **Traffic density**: 1.5 (expert difficulty)
- **Vehicle count**: 50 total on road
- **Lanes**: 4 (highway-v0 standard)
- **Observation**: Kinematics (position, velocity, heading)

---

## Validation Checklist

Before submitting videos:

- [ ] Video file exists and plays in VLC/media player
- [ ] Speedometer overlay visible and updates smoothly
- [ ] Colors change (green/red) based on speed zone
- [ ] Episode runs full 60 seconds without crashes (if possible)
- [ ] File size reasonable (~5-10 MB)
- [ ] Frame rate consistent throughout
- [ ] Model name in filename matches training outputs

---

## Summary

| Feature | Low-Freq Script | High-Freq Script |
|---|---|---|
| **Models** | E, F, G, H, K, L, M | I, J |
| **Default Model** | F (Best baseline) | I (Aggressive) |
| **Decision Rate** | 1 Hz | 5 Hz |
| **FPS** | 15 | 5 |
| **Speedometer** | ✓ | ✓ |
| **Duration** | 60s | 60s |
| **Use Case** | Quality comparison | Frequency analysis |
| **Best For** | Demos, comparisons | Analysis, validation |

Both scripts are **production-ready** and can be customized extensively for different models and environments.
