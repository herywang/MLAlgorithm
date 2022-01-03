import torch
import torch.nn as nn
import torch.nn.functional as F

from common import init_weights

MAX_LOG_STD = 20
MIN_LOG_STD = -2
EPSILON = 1e-6


class GaussianPolicyNetwork(nn.Module):
    '''
    SAC算法的策略网络我们又两种实现,一个是GaussianPolicy,
    加了最大熵,来使得action更加随机,即: 有多条最优策略时,
    随机的选择最优策略其中的一条策略
    '''

    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(GaussianPolicyNetwork, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_ouput = nn.Linear(hidden_dim, num_actions)
        self.log_std = nn.Linear(hidden_dim, num_actions)
        self.apply(init_weights)

        if action_space is None:
            # aciton的范围
            self.action_scale = torch.tensor(1., torch.float32)
            # action的中值
            self.action_bias = torch.tensor(0., torch.float32)
        else:
            self.action_scale = torch.FloatTensor((action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor((action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_ouput(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, min=MIN_LOG_STD, max=MAX_LOG_STD)
        return mean, log_std

    def sample(self, state):
        '''
        :param state: current environment of agent.
        :return:
            action: 采样的行为
            log_prob:
        '''
        mean, log_std = self.forward(state)  # 输出action的分布, 均值和对数标准差
        std = log_std.exp()  # 对数标准差取指数, 就得到标准差
        # 构建一个action高斯分布 action ~ N(mean, std)
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # 根据action随机的采样一个action
        y_t = torch.tanh(x_t)   # 将采样到的action缩放到 (0, 1)之间
        action = y_t * self.action_scale + self.action_bias # 对action进行rebuild, 获得到最终的action
        log_prob = normal.log_prob(x_t) # 计算此重采样后的action的对数概率 (这里是一个负数)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + EPSILON)

        # mean 加了一个最大熵随机.
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


class DeterministicPolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(DeterministicPolicyNetwork, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, num_actions)

        # 创建一个大小为: (1, num_actions)的张量
        self.noise = torch.FloatTensor(num_actions)

        if action_space is None:
            self.action_scale = 1.
            self.action_bias = 0.
        else:
            self.action_scale = torch.FloatTensor((action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor((action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = F.tanh(self.mean(x)) * self.action_scale + self.action_bias
        return mean

    def sample(self, state):
        mean = self.forward(state)
        # 将self.noise创建一个服从 N(0, 0.1)的分布采样数据
        self.noise.normal_(0., std=0.1)
        self.noise = self.noise.clamp(-0.25, 0.25)
        action = mean + self.noise
        return action, torch.tensor(0.), mean
