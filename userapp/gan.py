# models.py

import torch
import torch.nn as nn

# Define RNN Model
class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=1):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_dim)
        out, _ = self.rnn(x)  # out shape: (batch_size, sequence_length, hidden_dim)
        out = out[:, -1, :]    # Get the last time step's output
        out = self.fc(out)     # Apply the linear layer
        return out

# Define Generator Model
class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
            nn.Tanh()  # Output layer with Tanh activation
        )

    def forward(self, x):
        return self.fc(x)

# Define Discriminator Model
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()  # Output layer with Sigmoid activation
        )

    def forward(self, x):
        return self.fc(x)
