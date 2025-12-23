# Risk-Sensitive Navigation & Multi-Agent Interaction in Highway-Env

This repository contains the coursework implementation for **Deep Reinforcement Learning (CW2)**. The project investigates the safety benefits of Distributional RL (QR-DQN) compared to standard Risk-Neutral RL (DQN) in autonomous driving scenarios.

## 📌 Project Overview
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