import numpy as np
import pandas as pd
import random
from tqdm import tqdm

# Define reward levels to test for digit 2
reward_values_digit_2 = [1, 2, 4]
fixed_reward = 1  # Fixed reward for digits 5 and 8
digit_set = [2, 5, 8]

# Store all results
all_results = []

# Define the softmax choice model
def softmax_selection(preferences):
    exp_values = np.exp(preferences)
    probabilities = exp_values / np.sum(exp_values)
    return np.random.choice(len(preferences), p=probabilities)

# Simulate trials for each reward value for digit 2
for reward_2 in tqdm(reward_values_digit_2, desc="Running Multi-Digit Experiment"):
    rewards = {2: reward_2, 5: fixed_reward, 8: fixed_reward}
    counts = {d: 0 for d in digit_set}
    total_rewards = {d: 0 for d in digit_set}
    preferences = np.array([0.0, 0.0, 0.0])  # initial preferences

    learning_rate = 0.1
    n_trials = 1000

    for _ in range(n_trials):
        digit_idx = softmax_selection(preferences)
        selected_digit = digit_set[digit_idx]
        reward = rewards[selected_digit]

        # Update counts and rewards
        counts[selected_digit] += 1
        total_rewards[selected_digit] += reward

        # Update preference
        preferences[digit_idx] += learning_rate * (reward - preferences[digit_idx])

    # Compute selection rates
    selection_rates = {f'selection_rate_digit_{d}': counts[d]/n_trials for d in digit_set}
    total_reward = sum(total_rewards.values())

    # Compute reward ratios and response ratios
    def safe_ratio(a, b):
        return a / b if b != 0 else np.nan

    reward_ratios = {
        'reward_ratio_2vs5': safe_ratio(rewards[2], rewards[5]),
        'reward_ratio_2vs8': safe_ratio(rewards[2], rewards[8]),
    }

    response_ratios = {
        'response_ratio_2vs5': safe_ratio(counts[2], counts[5]),
        'response_ratio_2vs8': safe_ratio(counts[2], counts[8]),
    }

    result = {
        f'reward_value_digit_{d}': rewards[d] for d in digit_set
    }
    result.update(selection_rates)
    result['total_reward'] = total_reward
    result.update(reward_ratios)
    result.update(response_ratios)

    all_results.append(result)

# Convert to DataFrame and save
results_df = pd.DataFrame(all_results)
results_df.to_csv("multi_digit_matching_law_results.csv", index=False)
print("Saved results to multi_digit_matching_law_results.csv")
