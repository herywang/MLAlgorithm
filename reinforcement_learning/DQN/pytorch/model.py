import torch.nn as nn
from torch.nn import Module


class Net(Module):
    def __init__(self, n_features, n_actions, hidden1=128, hidden2=128):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_features, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.output = nn.Linear(hidden2, n_actions)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, states):
        x = self.fc1(states)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.output(x)
        x = self.relu(x)
        return x
