import torch
import torch.nn as nn
import torch.nn.functional as F

from common import init_weights


class CriticNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim=128):
        super(CriticNetwork, self).__init__()

        # Q1 architecture.
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture.
        self.linear4 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)

        self.apply(init_weights)

    def forward(self, state, action):
        inputs = torch.cat([state, action], dim=1)

        x1 = F.relu(self.linear1(inputs))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = F.relu(self.linear4(inputs))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)
        return x1, x2
