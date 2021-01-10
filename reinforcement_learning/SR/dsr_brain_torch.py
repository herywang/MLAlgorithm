import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import torchvision
import numpy as np
from dsr_maze_env import DSRMaze


class AutoEncoder(nn.Module):
    def __init__(self, feature_size, action_size, phi_size=4, buffer_size=100, batch_size=32):
        super(AutoEncoder, self).__init__()
        self.feature_size = feature_size  # states' feature dimension.
        self.action_size = action_size
        self.phi_size = phi_size
        self.count = 0
        self.batch_size = batch_size

        self.encoder = nn.Sequential(
            nn.Linear(self.feature_size, 8),
            nn.ReLU(),
            nn.Linear(8, 8),
            nn.ReLU(),
            nn.Linear(8, self.phi_size)
        )

        self.decoder = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 8),
            nn.ReLU(),
            nn.Linear(8, self.feature_size)
        )

    def forward(self, x):
        # 自动编码器训练部分
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


class Agent():
    def __init__(self):
        self.env = DSRMaze()
        self.count = 0
        self.buffer_size = 100
        self.replay_buffer = np.zeros(
            (self.buffer_size, self.env.feature_size * 2 + 2), dtype=float)
        self.batch_size = 8

    def store_transition(self, s, a, r, s_):
        index = self.count % self.buffer_size
        self.replay_buffer[index] = np.hstack((s, [a, r], s_))
        self.count += 1

    def get_train_data_set(self):
        choice_index = np.random.choice(
            self.buffer_size if self.buffer_size < self.count else self.count, self.batch_size)
        return self.replay_buffer[choice_index]

    def play(self):
        for i in range(100):
            s = self.env.get_current_state()
            action_index = np.random.randint(0, 4, 1)
            s_, reward, done = self.env.step(action_index[0])
            self.store_transition(s, action_index[0], reward, s_)
            if done:
                self.env.reset()

        autoEncoder = AutoEncoder(self.env.feature_size, self.env.action_size)
        optimizer = torch.optim.Adam(autoEncoder.parameters, 0.005)
        loss_func = nn.MSELoss()
        train_data = self.replay_buffer[:, 0: self.env.feature_size]
        for epoch in range(1000):
            b_x = train_data[:, 0:self.env.feature_size]
            b_y = train_data[:, 0:self.env.feature_size]
            encoded, decoded = autoEncoder(b_x)
            loss = loss_func(decoded, b_y)
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            print('Epoch:', epoch, ' | train loss: %.4f' % loss.data.numpy())


if __name__ == '__main__':
    agent = Agent()
    agent.play()
