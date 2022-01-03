import numpy as np
import torch
from torch.nn import Module
import torch.nn as nn
torch.random.manual_seed(0)

def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    init_data = torch.empty(size, dtype=torch.float32, requires_grad=True)
    return nn.init.uniform_(init_data, -v, v)


class Net(Module):
    def __init__(self, nb_states, nb_actions, hidden1=256, hidden2=256, init_w=3e-4):
        super(Net, self).__init__()
        # [states, goals] as input.
        self.fc1 = nn.Linear(nb_states * 2, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, nb_actions)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.3)
        self.init_weights(init_w)

    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.tanh(out)
        return out
