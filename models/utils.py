import numpy as np
import torch

# Set the random seed for reproducibility
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Seed set to {seed}")

# Calculate reward given a chosen action (digit)
def calculate_reward(digit, reward_structure):
    """
    Calculate reward based on the digit selected.
    :param digit: The digit chosen (either 2, 5, or 8)
    :param reward_structure: Dictionary mapping digits to their reward values
    :return: Reward value
    """
    return reward_structure.get(digit, 0)

# Convert preferences to action selection using a softmax approach
def softmax_selection(preferences):
    exp_values = np.exp(preferences)
    probabilities = exp_values / np.sum(exp_values)
    return np.random.choice(len(preferences), p=probabilities)

# Convert model outputs to action selection (softmax over preferences)
def select_action_from_model(model_output):
    probabilities = torch.softmax(model_output, dim=-1)
    action = torch.multinomial(probabilities, 1)
    return action.item()
