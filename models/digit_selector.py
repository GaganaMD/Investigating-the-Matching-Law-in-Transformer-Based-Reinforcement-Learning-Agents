import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple neural network for digit selection based on preferences
class DigitSelectorNN(nn.Module):
    def __init__(self):
        super(DigitSelectorNN, self).__init__()
        self.fc1 = nn.Linear(3, 64)  # Input layer for 3 digits
        self.fc2 = nn.Linear(64, 3)  # Output layer for 3 choices (2, 5, 8)

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # Apply ReLU activation
        x = self.fc2(x)  # Output layer
        return x

# Function to initialize the neural network
def initialize_model():
    model = DigitSelectorNN()
    return model

# Optimizer and Loss function
def get_optimizer(model, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    return optimizer

def get_loss_fn():
    return nn.CrossEntropyLoss()
