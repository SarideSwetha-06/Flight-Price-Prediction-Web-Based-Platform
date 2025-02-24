import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),  # Example hidden layer
            nn.ReLU(),
            nn.Linear(128, output_dim)  # Output layer
        )

    def forward(self, x):
        return self.fc(x)
