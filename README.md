# Investigating-the-Matching-Law-in-Transformer-Based-Reinforcement-Learning-Agents

# 🧠 Matching Law in Transformer-Based RL Agents

A Transformer-based reinforcement learning agent trained on a sequential decision-making task to investigate whether its behavior follows the **Matching Law** — a principle from behavioral psychology stating that organisms match their response rates to relative reward rates.

---

## 📌 Features

- 🔁 Two-choice sequential decision task with variable reward contingencies  
- 🤖 Transformer-based agent trained using deep Q-learning  
- 📈 Empirical analysis of choice behavior vs. reward ratios  
- 🧠 Insights into links between artificial and biological learning behavior  

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- PyTorch
- NumPy
- Matplotlib
- scikit-learn

Install dependencies:
```bash
pip install -r requirements.txt
```

### Run Training

```bash
python train_agent.py --reward_ratios 1 2 4 8 16
```

### Run Evaluation

```bash
python evaluate_agent.py --plot
```

---

## 📂 Project Structure

```
.
├── data/                    # MNIST-based visual stimuli
├── models/                  # Trained Transformer checkpoints
├── train_agent.py           # RL training loop
├── evaluate_agent.py        # Evaluation and plotting
├── utils.py                 # Helper functions
├── results/                 # Plots and logs
└── README.md
```

---

## 📊 Key Results

- Transformer agents **match response rates** to reward contingencies in most conditions  
- Slight **undermatching** occurs at high reward ratios (e.g. 16:1)  
- Log-log plots confirm near-linear relationship (slope ≈ 1) between response and reward ratios

---

## 🧪 Method Summary

- **Environment**: Two-alternative forced-choice using digit cues (e.g., 5 vs 2)  
- **Model**: Transformer encoder with attention layers  
- **Learning**: Deep Q-learning using temporal-difference updates  
- **Evaluation**: Response ratios, reward ratios, total rewards, log-log plots

---

## 📖 References

- [Decision Transformer](https://arxiv.org/abs/2106.01345)  
- [Matching Law – Herrnstein (1961)](https://doi.org/10.1901/jeab.1961.4-267)  
- [DQN – Mnih et al. (2015)](https://www.nature.com/articles/nature14236)  
- [NeuroFarm](https://github.com/mariakesa/NeuroFarm)

---

## 🙌 Acknowledgements

Thanks to **Maria Kesa** and **Saqar Sabzali** for foundational experiments, **Zahra Sarayloo** for mentorship, and the **Neuromatch** and **Impact Scholars** teams for support.

---
