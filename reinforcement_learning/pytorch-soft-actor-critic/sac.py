import os
import random

import numpy as np
import torch
from torch.nn import functional as F
from torch.optim import Adam

from actor_model import GaussianPolicyNetwork
from common import hard_update, soft_update
from critic_model import CriticNetwork


class ReplayMemory:
    def __init__(self, capacity, seed=1):
        self.capacity = capacity
        self.buffer = []
        random.seed(seed)
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[(self.position + 1) % self.capacity] = (state, action, reward, next_state, done)
        self.position += 1

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


class SAC:
    def __init__(self, num_inputs, action_space, args):
        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha
        self.batch_size = args.batch_size
        self.automatic_entropy_tuning = args.automatic_entropy_tunning
        self.target_update_interval = args.target_update_interval
        self.lr = args.lr

        self.action_space = action_space

        self.replay_buffer = ReplayMemory(5000)

        self.critic = CriticNetwork(num_inputs, action_space.shape[0], args.hidden_size)
        self.critic_opt = Adam(self.critic.parameters(), lr=self.lr)

        self.critic_target = CriticNetwork(num_inputs, action_space.shape[0], args.hidden_size)
        hard_update(self.critic_target, self.critic)
        self.target_entroy = -torch.prod(torch.Tensor(action_space.shape)).item()
        self.policy = GaussianPolicyNetwork(num_inputs, action_space.shape[0], args.hidden_size, action_space)
        self.policy_optim = Adam(self.policy.parameters(), lr=self.lr)

        self.update_num = 0

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).unsqueeze(0)  # 1 dimention to 2 dimentions.
        if evaluate:
            # 如果是评估, 使用最大熵的方式,增加action的随机探索度, 即: 多峰最优
            _, _, action = self.policy.sample(state)
        else:
            # 否则直接输出策略的action输出
            action, _, _ = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]

    def learn(self):
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states_batch = torch.FloatTensor(states)
        actions_batch = torch.FloatTensor(actions)
        rewards_batch = torch.FloatTensor(rewards).unsqueeze(1)
        dones_batch = torch.FloatTensor(dones).unsqueeze(1)
        next_states_batch = torch.FloatTensor(next_states)

        # 使用了两个Q函数相加作为真实的Q结果. (论文中算法)
        qf1, qf2 = self.critic(states_batch, actions_batch)

        # 计算next q value
        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_states_batch)
            qf1_next, qf2_next = self.critic_target(next_states_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next, qf2_next) - self.alpha * next_state_log_pi
            next_q_value = rewards_batch + dones_batch * self.gamma * min_qf_next_target

        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        # 训练critic网络
        self.critic_opt.zero_grad()
        qf_loss.backward()
        self.critic_opt.step()

        # 训练actor网络
        pi, log_pi, _ = self.policy.sample(states_batch)
        qf1_pi, qf2_pi = self.critic(states_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        # policy网络优化采用梯度上升方式, 这里已经对loss取了相反数
        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()
        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        self.update_num += 1

        # 如果开启了自动调节α
        if self.automatic_entropy_tuning:
            self.target_entropy = -torch.prod(torch.Tensor(self.action_space.shape)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True)
            self.alpha_optim = Adam([self.log_alpha], lr=self.lr)
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = self.log_alpha.exp()

        if self.update_num % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

    def save_model(self, env_name, suffix="", actor_path=None, critic_path=None):
        if not os.path.exists("./models/"):
            os.mkdir("./models")
        if actor_path is None:
            actor_path = "models/sac_actor_{}_{}".format(env_name, suffix)
        if critic_path is None:
            critic_path = "models/sac_critic_{}_{}".format(env_name, suffix)
        print("Saving models to {} and {}".format(actor_path, critic_path))
        torch.save(self.policy.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

    def load_model(self, actor_path, critic_path):
        '''
        Load model.
        :param actor_path
        :param critic_path:
        :return:
        '''
        print("Loading model from {} and {}".format(actor_path, critic_path))
        if actor_path is not None:
            self.policy.load_state_dict(torch.load(actor_path))
        if critic_path is not None:
            self.critic.load_state_dict(torch.load(critic_path))
