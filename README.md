# Investigating the Matching Law in Transformer-Based RL Agents

Code from a Neuromatch Impact Scholars project. The question: does a transformer-based reinforcement-learning agent split its choices between two options in proportion to their rewards, as the Matching Law from behavioural psychology predicts (Herrnstein, 1961)?

**Report:** Y. Karakoc, E. Sharifi, M. D. Gagana, Z. Sarayloo. *Investigating the Matching Law in Transformer-Based Reinforcement Learning Agents.* Zenodo, 2025. DOI [10.5281/zenodo.15462678](https://doi.org/10.5281/zenodo.15462678). The record also includes the figure code and a CRediT contribution statement.

The report is the reference for the method and results. This repository holds the experiment and analysis code; it does not include trained models or result files.

## Task

Each trial shows an MNIST digit cue (for example a 2 or a 5), and the agent chooses one of the two options. The reward for one digit is fixed at 1. The reward for the other varies across runs (the main script uses 1, 2, 4, 8 and 16). After training, the agent's response ratio is compared with the reward ratio on a log-log plot. A slope near 1 means matching.

## Repository contents

| File | What it does |
|---|---|
| `experiment_1_transformer.py` | Main experiment. A gym environment of MNIST digit pairs, and a transformer encoder (3 layers, 8 heads, d_model 64) used as the feature extractor of a PPO policy (Stable-Baselines3). Runs 5 seeds per reward value for 5,000 steps each and writes `matching_law_results.csv`. |
| `experiments/experiment1_binary_digits.py` | Smaller supervised two-digit variant (digit-2 reward of 1, 2 or 4). |
| `experiments/experiment2_multi_digits.py` | Three-option variant (digits 2, 5 and 8). Writes `multi_digit_matching_law_results.csv`. |
| `analysis/visualize_results.py` | Log-log matching plot from `matching_law_results.csv`. |
| `analysis/compare_experiments.py` | Compares the two-option and three-option results. |
| `analysis/reward_pie_charts.py` | Reward share per option. |
| `plots.py` | Combined plotting script used for the report figures. |
| `models/` | Small helper network and utilities (seeding, selection rate, reward ratio). |

## Running it

```bash
pip install -r requirements.txt
pip install torchvision "gym<0.26" "stable-baselines3<2.0"   # needed by experiment_1_transformer.py
python experiment_1_transformer.py        # writes matching_law_results.csv and learning_curves.png
mkdir -p results
python analysis/visualize_results.py      # writes results/matching_law_results.png
```

MNIST downloads automatically through torchvision on the first run.

## Known issues

These are recorded so nobody mistakes this code for a finished, packaged release:

- **The training algorithm differs from the report.** The report describes a deep Q-learning variant. `experiment_1_transformer.py` trains the transformer policy with PPO. Use the report's description for the published results.
- **`experiments/experiment1_binary_digits.py` won't import as written.** It imports `DigitSelectorNet`, but `models/digit_selector.py` defines `DigitSelectorNN`.
- **Output paths aren't consistent.** Some scripts write to the repository root and others to `results/` or `../results/`, so run them from the root and create `results/` first.
- **No saved results.** Trained models and result CSVs aren't committed. The figures in the report come from the authors' runs.

## Credits

Study led by Yagmur Karakoc and Zahra Sarayloo, with Emad Sharifi and Gagana M D contributing (see the CRediT statement in the Zenodo record). Foundational experiments by Maria Kesa and Saqar Sabzali ([NeuroFarm](https://github.com/mariakesa/NeuroFarm)). Thanks to the Neuromatch Impact Scholars program.
