# experiment1_binary_digits.py

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import pandas as pd
import os

from models.digit_selector import DigitSelectorNet
from models.utils import set_seed, compute_selection_rate, compute_reward_ratio

set_seed(42)

# -----------------------------
# Parameters
# -----------------------------
target_digits = [2, 5]
rewards = {2: [1, 2, 4], 5: 1}  # Reward values for digit 2
num_epochs = 3
batch_size = 64
learning_rate = 0.001

# -----------------------------
# Data Loading
# -----------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

mnist_train = datasets.MNIST(root="../data/raw", train=True, download=True, transform=transform)

def filter_digits(dataset, digits):
    indices = [i for i, (_, label) in enumerate(dataset) if label in digits]
    return Subset(dataset, indices)

data = filter_digits(mnist_train, target_digits)
data_loader = DataLoader(data, batch_size=batch_size, shuffle=True)

# -----------------------------
# Training + Evaluation
# -----------------------------
all_results = []

for reward_val in rewards[2]:
    model = DigitSelectorNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    label_rewards = {2: reward_val, 5: rewards[5]}

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        for images, labels in data_loader:
            mask = (labels == 2) | (labels == 5)
            images, labels = images[mask], labels[mask]
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Evaluation
    model.eval()
    correct_by_digit = {2: 0, 5: 0}
    total_by_digit = {2: 0, 5: 0}
    total_reward = 0

    with torch.no_grad():
        for images, labels in data_loader:
            mask = (labels == 2) | (labels == 5)
            images, labels = images[mask], labels[mask]
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            for i, label in enumerate(labels):
                pred = predicted[i].item()
                actual = label.item()
                if pred == actual:
                    correct_by_digit[actual] += 1
                    total_reward += label_rewards[actual]
                total_by_digit[actual] += 1

    selection_rate_2 = correct_by_digit[2] / (correct_by_digit[2] + correct_by_digit[5])
    reward_ratio = compute_reward_ratio(label_rewards[2], label_rewards[5])
    response_ratio = selection_rate_2 / (1 - selection_rate_2)

    all_results.append({
        "reward_value_digit2": label_rewards[2],
        "selection_rate_digit2": selection_rate_2,
        "reward_ratio": reward_ratio,
        "response_ratio": response_ratio,
        "total_reward": total_reward
    })

# Save results
os.makedirs("../results", exist_ok=True)
pd.DataFrame(all_results).to_csv("../results/matching_law_results.csv", index=False)
print("Experiment 1 completed and results saved.")
