import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

MAX_EPISODE = 1000
RENDER = True


class DiscretePolicyGradient(nn.Module):
    def __init__(self, n_features, n_actions, num_hiddens=64):
        super(DiscretePolicyGradient, self).__init__()
        self.linear1 = nn.Linear(n_features, num_hiddens)
        self.linear2 = nn.Linear(num_hiddens, num_hiddens)
        self.output = nn.Linear(num_hiddens, n_actions)

    def forward(self, state):
        x = F.tanh(self.linear1(state))
        x = F.tanh(self.linear2(x))
        x = F.softmax(self.output(x), dim=1)
        return x


class ReplayBuffer:
    def __init__(self):
        self.replay_buffer = []
        self.count = 0

    def store_transition(self, s, a, r):
        self.replay_buffer.append(np.array([s, a, r]))
        self.count += 1

    def get_observations(self):
        return np.vstack(np.vstack(self.replay_buffer)[:, 0])

    def get_actions(self):
        return np.vstack(np.vstack(self.replay_buffer)[:, 1])

    def clear(self):
        self.replay_buffer = []
        self.count = 0

    def get_reward(self, i):
        return np.vstack(self.replay_buffer)[i, 2]


class Agent:
    def __init__(self, n_features, n_actions, num_hiddens=64):
        self.n_features = n_features
        self.n_actions = n_actions
        self.model = DiscretePolicyGradient(n_features, n_actions, num_hiddens)
        self.opt = optim.Adam(self.model.parameters())
        self.replay_buffer = ReplayBuffer()

        self.gamma = 0.9
        self.writer = SummaryWriter("./torch_cartpole_log")

    # 根据当前观察选择一个action
    def choose_action(self, observation):
        if not isinstance(observation, torch.Tensor):
            observation = torch.FloatTensor(observation)
        if observation.dim() == 1:
            observation = observation.unsqueeze(0)
        # 关闭梯度计算
        with torch.no_grad():
            action_prob = self.model(observation)
        c = Categorical(action_prob)
        action_index = c.sample()[0]
        return action_index

    # 执行一次学习
    def learn(self):
        discounted_reward_norm: np.ndarray = self.__discount_and_norm_rewards()
        observations = torch.FloatTensor(self.replay_buffer.get_observations())
        actions = torch.LongTensor(self.replay_buffer.get_actions())
        self.model.train()
        acts_prob = self.model(observations)
        # 这里需要注意: torch求梯度只能对标量求,不能对向量求解
        loss = F.binary_cross_entropy(acts_prob,
                                      torch.autograd.Variable(F.one_hot(actions.squeeze(1), self.n_actions).type(torch.float32)),
                                      torch.from_numpy(discounted_reward_norm).unsqueeze(1))
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        # 一次学习过后, 清空replay_buffer中的全部数据
        self.replay_buffer.clear()
        return np.sum(discounted_reward_norm), discounted_reward_norm.shape[0]

    # 将reward进行折扣
    def __discount_and_norm_rewards(self):
        size = self.replay_buffer.count
        discounted_rewards = np.zeros((size))
        running_add = 0
        for i in range(self.replay_buffer.count - 1, -1, -1):
            running_add = running_add * self.gamma + self.replay_buffer.get_reward(i)
            discounted_rewards[i] = running_add
        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= np.std(discounted_rewards)
        return discounted_rewards


def run():
    env = gym.make('CartPole-v0')
    env.seed(1)
    n_actions = env.action_space.n
    n_features = env.observation_space.shape[0]
    agent = Agent(n_features, n_actions)
    for episode in range(MAX_EPISODE):
        observation = env.reset()
        while True:
            if RENDER:
                env.render()
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0)
            action = agent.choose_action(obs_tensor).cpu().numpy()
            next_observation, reward, done, info = env.step(action)
            agent.replay_buffer.store_transition(observation, action, reward)
            observation = next_observation
            if done:
                sum_reward, play_time = agent.learn()
                agent.writer.add_scalar("sum_reward", sum_reward, episode)
                agent.writer.add_scalar("play_time", play_time, episode)
                break
    agent.writer.close()
    torch.save(agent.model, "./trained_model/whole_model.pt")
    torch.save(agent.model.state_dict(), "./trained_model/state_dict.pt")

    env.close()


if __name__ == '__main__':
    run()
