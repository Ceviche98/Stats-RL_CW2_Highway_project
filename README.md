# Risk-Sensitive Navigation & Multi-Agent Interaction in Highway-Env

This repository contains the coursework implementation for **Deep Reinforcement Learning (CW2)**. The project investigates the safety benefits of Distributional RL (QR-DQN) compared to standard Risk-Neutral RL (DQN) in autonomous driving scenarios.

## 📌 Project Overview

This project evaluates DQN and QR-DQN agents in high-density highway traffic (ρ=1.5) under four configurations, with comprehensive robustness testing and generalization analysis.

### Key Findings
- **Best Model**: F (QR-DQN Aggressive, Low-Freq) - **13.8% crash rate** ⭐
- **Decision Quality > Decision Frequency**: 5 Hz models underperform despite faster updates
- **QR-DQN superiority**: 5.2x safer than DQN baseline (F vs E)
- **Generalization**: Models trained on highway transfer to merge/roundabout tasks

---

## 📁 Quick Navigation

| Experiment | Purpose | Scripts | Time |
|---|---|---|---|
| **Exp1** | Train baseline models (13 total A-M) | `7_*.py` (A-D), `6_*.py` (E-H), `8_*.py` (I-J), `9_*.py` (K-M) | 8-12 hours per model |
| **Exp2** | Robustness (latency/noise/blackout) | `exp2/test_*.py` | 1-2 hours |
| **Exp3** | Generalization & transfer learning | `exp3/0_*.py`, `evaluate_*.py` | 6-8 hours |
| **Videos** | Qualitative visualization | `record_video_*.py` | 5 min |

---

## ⚡ Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Verify setup
python -c "import gymnasium; import highway_env; print('✓ Ready')"
```

---

## 🎯 Experiment 1: Baseline Training & TABLE III

### Overview
Train DQN/QRDQN under 4 configurations to populate TABLE III (13 models, A-M).

**Configurations:**
- Frequency: **High (5 Hz)** vs **Low (1-2 Hz)**
- Mode: **Aggressive** (0.5 speed reward) vs **Conservative** (0.3 speed reward)
- Algorithm: **DQN** vs **QRDQN**

### How to Reproduce TABLE III

#### Step 0: High-Frequency Baseline (Models A, B, C, D)
```bash
python experiments/exp1/7_train_highf.py
```
**Generates:**
- `A_High_freq_aggressive_DQN_s200` - DQN baseline, aggressive mode
- `B_High_freq_aggressive_QRDQN_s200` - QRDQN aggressive, high-freq variant
- `C_High_freq_conservative_DQN_s200` - DQN baseline, conservative mode
- `D_High_freq_conservative_QRDQN_s200` - QRDQN conservative, high-freq variant

**What it does:**
- Creates 5 Hz control baseline (non-extended, standard training)
- Two reward modes: aggressive (0.5 speed, -15 collision) and conservative (0.3 speed, -40 collision)
- Standard training: 750K steps (half the extended version)
- Larger observation window: x ∈ [-100, 100], y ∈ [-100, 100]
- **Time: ~60 minutes total (30 min per mode pair)**

**Key difference from Step 3:**
- Step 0 (`7_train_highf.py`): **Standard 750K steps, baseline 5 Hz**
- Step 3 (`8_train_highf_final.py`): **Extended 3.75M steps, optimized 5 Hz** (5x longer training)
- Models I, J (Step 3) are extended versions, typically better but sometimes overfitting

---

#### Step 1: Low-Frequency Aggressive (Models E, F)
```bash
python experiments/exp1/6_train_aggressive.py
```
**Generates:**
- `E_Low_freq_aggr_DQN_s200` - 67.2% crash rate
- `F_Low_freq_aggr_QRDQN_s200` - **13.8% crash rate** ⭐ BEST

**What it does:**
- Creates 1 Hz control environments (1 decision per second)
- Aggressive rewards: 0.5 speed bonus, -15 collision penalty
- 750K steps training with multiprocessing
- **Time: ~30 minutes**

**Customize by editing function:**
```python
train_experiment(model_type="QRDQN", seed=200)  # seed controls randomness
```

---

#### Step 2: Low-Frequency Conservative (Models G, H)
```bash
python experiments/exp1/6_train_conservative.py
```
**Generates:**
- `G_Low_freq_cons_DQN_s200` - 98.2% crash rate
- `H_Low_freq_cons_QRDQN_s200` - 36.6% crash rate

**What it does:**
- 1 Hz control, Conservative rewards: 0.3 speed, -40 collision penalty
- Tests safety-first paradigm
- **Paradoxical finding**: Heavy penalties ≠ safer (models become too timid)
- **Time: ~30 minutes**

---

#### Step 3: High-Frequency Models (Models I, J)
```bash
python experiments/exp1/8_train_highf_final.py
```
**Generates:**
- `I_High_freq_aggressive_QRDQN_s500_opt` - 30.4% crash rate
- `J_High_freq_conservative_QRDQN_s500_opt` - 23.8% crash rate

**What it does:**
- Creates 5 Hz control (5 decisions/second = 0.2s per decision)
- Extended training: 3.75M steps (5x longer)
- Smaller observation window: x ∈ [-100, 100]
- **Key finding**: Despite faster decisions, worse than low-freq model F!
- **Time: ~60 minutes**

**Why high-freq models perform worse:**
- Reward tuning optimized for low-freq behavior
- High-freq requires different hyperparameters
- Demonstrates decision quality matters more than frequency

---

#### Step 4: Optimized Low-Frequency Variants (Models K, L, M)
```bash
python experiments/exp1/9_train_low_aggressive.py
```
**Generates:**
- `K_Low_freq_aggressive_QRDQN_s300_opt` - 24.5% crash rate
- `L_Low_freq_aggressive_QRDQN_s300_opt` - 42.0% crash rate
- `M_Low_freq_aggressive_QRDQN_s300_opt` - 26.8% crash rate

**What it does:**
- Three variants with tuned hyperparameters
- All use QRDQN + 1 Hz frequency except K
- Vary: γ (0.93), collision_reward (-13), observation range ([-150,150])
- Extended training: 3M steps
- **Time: ~120 minutes**

**Variants explained:**
| Model | γ | R_col | Special | Notes |
|---|---|---|---|---|
| K | 0.93 | -13 | see_behind + expanded obs | Base optimized |
| L | 0.93 | -13 | Same as K | Freq: 2 Hz baseline |
| M | 0.93 | -13 | Baseline obs | +see_behind+obs range |

**Customize any variant:**
```python
train_low_freq(
    model_type="QRDQN",
    mode="aggressive",
    seed=300,
    model_index="K",
    gamma=0.93,              # Discount factor (0.90, 0.95, 0.99)
    collision_reward=-13,    # Crash penalty (-10, -15, -20)
    policy_freq=2,           # Decisions/sec (1, 2, 5)
    obs_range_x=[-150, 150]  # Observation distance (±100, ±150, ±200)
)
```

---

### Generated Models Summary

| Model | Type | Config | Crash% | Notes |
|---|---|---|---|---|
| A | High/Agg/DQN | Baseline | 99.9 | Poor |
| B | High/Agg/QRDQN | Baseline | 20.0 | Better but slow |
| C | High/Cons/DQN | Baseline | 99.4 | Poor |
| D | High/Cons/QRDQN | Baseline | 30.7 | Conservative |
| E | Low/Agg/DQN | Baseline | 67.2 | Baseline |
| **F** | **Low/Agg/QRDQN** | **Baseline** | **13.8%** | **⭐ BEST** |
| G | Low/Cons/DQN | Baseline | 98.2 | Over-conservative |
| H | Low/Cons/QRDQN | Baseline | 36.6 | Conservative |
| I | High/Agg/QRDQN | Net:[512,512], γ:0.99 | 30.4 | Paradox: worse than F |
| J | High/Cons/QRDQN | Net:[512,512], γ:0.99 | 23.8 | Better than baseline |
| K | Low/Agg/QRDQN | γ:0.93, R_col:-13 | 24.5% | Optimized |
| L | Low/Agg/QRDQN | Freq: 2Hz | 42.0% | Freq test |
| M | Low/Agg/QRDQN | Obs: baseline | 26.8% | Obs range test |

---

## 🛡️ Experiment 2: Robustness Ablation (POMDP)

Evaluate best models (E, F) under perceptual degradation without retraining.

**Hypothesis:** QR-DQN's distributional approach provides superior risk management, but degrades predictably under adversarial conditions.

### Test 1: System Latency
Action execution delays of k ∈ {1, 2, 3, 4} frames.

```bash
python experiments/exp2/test_latency.py
```

**What it does:**
- Delays agent actions: 1 frame = 1 second at 1 Hz
- Tests planning under known future delay
- Output: `results/exp2_robustness/latency_results_*.csv`

**Customize:**
```python
DECISION_DELAYS = [0, 1, 2, 3, 4]  # Change range
N_EPISODES = 600                    # More episodes = more stable
```

---

### Test 2: Sensor Noise
Gaussian noise N(0,σ) added to vehicle coordinate observations.

```bash
python experiments/exp2/test_noise.py
```

**What it does:**
- Adds noise: x_obs = x_true + N(0,σ)
- Tests robustness to noisy perception (GPS/sensor error)
- Noise levels: σ ∈ {0.0, 0.025, 0.05, 0.075, 0.1, 0.15}
- Output: `results/exp2_robustness/noise_results_*.csv`

**Customize:**
```python
NOISE_LEVELS = [0.0, 0.025, 0.05, 0.075, 0.1, 0.15]
```

---

### Test 3: Sensor Blackout
Random observation dropout with probability p ∈ {0.0, 0.05, ..., 0.2}.

```bash
python experiments/exp2/test_blackout_gradient.py
```

**What it does:**
- Zeros out observation with probability p
- Simulates complete sensor failure
- Tests recovery from information gaps
- Output: `results/exp2_robustness/blackout_results_*.csv`

**Customize:**
```python
BLACKOUT_PROBS = [0.0, 0.05, 0.075, 0.1, 0.15, 0.2]
```

---

## 🚀 Experiment 3: Generalization & Transfer

Assess if agents learned robust dynamics or memorized training conditions.

### Phase 1: Density Stress Test
Increase traffic density ρ from 1.5→2.5, identify "break point" (crash rate >50%).

```bash
python experiments/exp3/0_evaluate_capacity_curriculum.py
```

**What it does:**
- Tests E, F, G, H at increasing densities
- Output: `results/capacity_curriculum_full_metrics.csv`
- Identifies maximum safe operating density
- **Time: 45 minutes**

---

### Phase 2: Train Merge Environment Models
Transfer learning on merge-v0 (lane merging) environment.

```bash
python experiments/exp3/1_train_merge.py
```

**What it does:**
- Trains DQN/QRDQN on merge task from scratch
- Generates: `exp3_merge_DQN_s500`, `exp3_merge_QRDQN_s500`
- Preparation for zero-shot transfer
- **Time: 60 minutes**

---

### Phase 3: Evaluate Generalization
Zero-shot transfer from highway/merge → roundabout.

```bash
# Generalization on highway at varying densities
python experiments/exp3/evaluate_transfer_density.py

# Transfer to roundabout (zero-shot)
python experiments/exp3/evaluate_transfer_metrics.py
```

**What it does:**
- Tests if learned "gap acceptance" logic transfers
- Output: Metrics, plots showing transfer success
- Validates algorithm learned abstract principles
- **Time: 60 minutes**

---

## 🎬 Video Generation

### Low-Frequency Video with Speedometer

Generate MP4 showing agent driving with real-time speedometer overlay.

```bash
python experiments/exp1/record_video_s200_speed.py
```

**Output:** `results/videos/speedometer_F_Low_freq_aggr_QRDQN_s200.mp4`

**Features:**
- 60-second episode
- Speedometer in top-left corner
- Color-coded: 🟢 Green (reward zone), 🔴 Red (too fast), 🔵 Blue (too slow)
- 15 FPS playback

**Customize - Try different models:**
```python
# Line 11-12
model_name = "E_Low_freq_aggr_DQN_s200"       # Try: E, F, G, H, K, L, M
MODEL_PATH = os.path.join(script_dir, "models/exp1/{}.zip".format(model_name))
```

**Customize - Change environment:**
```python
# Lines 16-41 - env_config dictionary
env_config = {
    "vehicles_density": 1.5,      # Try: 1.0, 2.0, 2.5
    "policy_frequency": 1,        # Try: 2, 5 (Hz)
    "collision_reward": -15,      # Try: -10, -20
    "high_speed_reward": 0.5,     # Try: 0.3, 0.7
    "reward_speed_range": [25, 35], # Try: [20, 30], [30, 40]
}
```

**Troubleshooting:**
- Model not found → Check `models/exp1/` structure
- Video corrupted → `pip install imageio-ffmpeg`
- Slow video → Reduce FPS or model complexity

---

### High-Frequency Video (5 Hz Models)

For models I, J (5 Hz):

```bash
python experiments/exp1/record_video_highf_s200_speed.py
```

**Differences:**
- `policy_frequency: 5` (5 decisions/second)
- More frequent updates = smoother video
- Output: `results/videos/debug_highfreq_run.mp4`

---

## 📊 Table III: Complete Results

| ID | Freq | Mode | Alg | Cr(%) | Rew | Spd | Std | Configuration | Notes |
|---|---|---|---|---|---|---|---|---|---|
| A | H | A | DQN | 99.9 | -5.8 | 29.6 | 17.6 | Baseline | |
| B | H | A | QR | 20.0 | 38.4 | 20.8 | 24.3 | Baseline | |
| C | H | C | DQN | 99.4 | -19.0 | 29.4 | 18.0 | Baseline | |
| D | H | C | QR | 30.7 | 26.4 | 20.6 | 30.8 | Baseline | |
| E | L | A | DQN | 67.2 | -8.4 | 23.5 | 7.6 | Baseline | |
| **F** | **L** | **A** | **QR** | **13.8** | **-2.1** | **20.0** | **5.2** | **Baseline** | **⭐ BEST** |
| G | L | C | DQN | 98.2 | -15.8 | 25.4 | 4.4 | Baseline | |
| H | L | C | QR | 36.6 | -7.2 | 19.9 | 9.7 | Baseline | |
| I | H | A | QR | 30.4 | 53.3 | 23.6 | 37.1 | Net: [512,512], γ: 0.99 | |
| J | H | C | QR | 23.8 | 33.6 | 21.4 | 29.1 | Net: [512,512], γ: 0.99 | |
| K | L | A | QR | 24.5 | -3.2 | 20.0 | 5.6 | γ: 0.93, Rcol: -13 | |
| L | L | A | QR | 42.0 | -4.3 | 21.4 | 6.6 | Freq: 2 Hz | |
| M | L | A | QR | 26.8 | -3.8 | 20.0 | 5.8 | Obs: Baseline | |

**Legend:**
- **Cr(%)**: Crash rate %
- **Rew**: Avg episode reward
- **Spd**: Avg speed (m/s)
- **Std**: Speed std deviation
- **H/L**: High (5 Hz) / Low (1-2 Hz) frequency
- **A/C**: Aggressive / Conservative mode
- **QR**: QR-DQN, **DQN**: Standard DQN

---

## 🔍 Troubleshooting

| Problem | Solution |
|---|---|
| Model not found | Run training scripts first; check `models/exp1/` |
| CUDA out of memory | Set `device="cpu"` or reduce batch_size |
| Video won't play | Install `imageio-ffmpeg`: `pip install imageio-ffmpeg` |
| Slow training | Verify multiprocessing active: check CPU usage |
| Import errors | Reinstall: `pip install --upgrade -r requirements.txt` |

---

## 📚 Key Insights

1. **QR-DQN >> DQN**: 5.2x safer (F vs E) due to risk-aware decisions
2. **Low-Freq > High-Freq**: 13.8% vs 30.4% (F vs I) - decision quality matters
3. **Conservative paradox**: Heavy penalties (-40) worsen performance (H vs F)
4. **Optimal tuning**: γ=0.93, R_col=-13 achieves 24.5% (K)
5. **Transfer works**: Merge-trained agents generalize to roundabout

---

## 📖 References

**Experiment 1 Context:**
> "We evaluate DQN and QR-DQN in a high-density highway scenario (ρ=1.5). We hypothesize QR-DQN's distributional approach enables superior risk management. The experiment varies Decision Frequency (High 5 Hz vs Low 1 Hz) and Reward Structure (Aggressive vs Conservative)."

**Experiment 2 Context:**
> "To evaluate agent reliability under partial observability, we subjected best-performing models to perceptual degradation without retraining. Three stressors: System Latency (k ∈ {1,2,...}), Sensor Noise (N(0,σ)), and Sensor Blackout (probability p)."

**Experiment 3 Context:**
> "This experiment assessed whether agents learned robust driving dynamics or merely memorized training conditions through density stress tests and zero-shot topology transfer. Density Stress Test: ρ increased 1.5→2.5. Topology Transfer: Zero-shot evaluation of merge-trained agents on roundabout-v0."

---

## 📝 How to Cite

If using this code, reference the experiment structure and models:

```bibtex
@misc{cw2_drl_highway,
  title={Risk-Sensitive Navigation with QR-DQN in Highway-Env},
  author={Your Name},
  year={2026},
  howpublished={\url{github.com/}}
}
```

---

**Last Updated:** January 2026
**Status:** ✅ Complete and tested
We address key challenges in Reinforcement Learning identified by *Arulkumaran et al. (2017)*, specifically **Safety**, **Partial Observability**, and **Multi-Agent Coordination**.

**Hypothesis:** A Risk-Sensitive agent (optimizing Conditional Value at Risk) will demonstrate superior robustness to sensor noise and aggressive traffic compared to a standard agent optimizing average reward.

### 🔬 Experiments
1.  **Baseline Efficiency:** Benchmarking "Cost of Safety" (Velocity vs. Crash Rate) on `highway-v0`.
2.  **Robustness (POMDP):** Ablation study testing resilience to **Sensor Noise**, **Lag**, and **Blackouts**.
3.  **Generalization:** Zero-shot transfer learning from `merge-v0` (training) to `roundabout-v0` (testing).
4.  **Adversarial Safety:** Survival rate against aggressive "Cut-In" agents.
5.  **Multi-Agent (The Broken Platoon):** Cooperative traffic smoothing to prevent rear-end collisions in a platoon.

---

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/highway-risk-project.git](https://github.com/YOUR_USERNAME/highway-risk-project.git)
   cd highway-risk-project