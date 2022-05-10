import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam

from model import (Actor, Critic)
from memory import SequentialMemory
from random_process import OrnsteinUhlenbeckProcess
from util import *

# from ipdb import set_trace as debug

lossFunction = nn.MSELoss()


class DDPG:
    def __init__(self, nb_states, nb_actions, args):
        """
        :param nb_states: number of states.
        :param nb_actions: number of actions.
        :param args: the other parameters.
        """

        if args.seed > 0:
            self.seed(args.seed)

        self.nb_states = nb_states
        self.nb_actions = nb_actions

        # Create Actor and Critic Network
        net_cfg = {'hidden1': args.hidden1, 'hidden2': args.hidden2, 'init_w': args.init_w}
        self.actor = Actor(self.nb_states, self.nb_actions, **net_cfg)
        self.actor_target = Actor(self.nb_states, self.nb_actions, **net_cfg)
        self.actor_optim = Adam(self.actor.parameters(), lr=args.prate)

        self.critic = Critic(self.nb_states, self.nb_actions, **net_cfg)
        self.critic_target = Critic(self.nb_states, self.nb_actions, **net_cfg)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.rate)

        hard_update(self.actor_target, self.actor)  # Make sure target is with the same weight
        hard_update(self.critic_target, self.critic)    # 创建完成网络之后就先copy一份，保证target_net和eval_net有相同的weight

        # Create replay buffer
        self.memory = SequentialMemory(limit=args.rmsize, window_length=args.window_length)
        self.random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=args.ou_theta, mu=args.ou_mu, sigma=args.ou_sigma)

        # Hyper-parameters
        self.batch_size = args.bsize
        # soft-update coefficient.
        self.tau = args.tau
        self.discount = args.discount
        self.depsilon = 1.0 / args.epsilon

        # 
        self.epsilon = 1.0
        self.s_t = None  # Most recent state
        self.a_t = None  # Most recent action
        self.is_training = True


    def update_policy(self):
        # Sample batch
        state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = self.memory.sample_and_split(self.batch_size)

        # Prepare for the target q batch
        with torch.no_grad():
            next_q_values = self.critic_target([to_tensor(next_state_batch), self.actor_target(to_tensor(next_state_batch))])

        target_q_batch = to_tensor(reward_batch) + self.discount * to_tensor(terminal_batch.astype(np.float)) * next_q_values

        # Critic update
        self.critic.zero_grad()
        q_batch = self.critic([to_tensor(state_batch), to_tensor(action_batch)])
        value_loss = lossFunction(q_batch, target_q_batch)
        # 计算出梯度
        value_loss.backward()
        # 执行优化
        self.critic_optim.step()

        # Actor update
        self.actor.zero_grad()
        actor_action_batch = self.actor(to_tensor(state_batch))
        policy_loss = -self.critic([to_tensor(state_batch), actor_action_batch])
        policy_loss = policy_loss.mean()
        # 计算出梯度
        policy_loss.backward()
        # 执行优化
        self.actor_optim.step()

        # Target update
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

    def eval(self):
        """
        训练完train样本后，生成的模型model要用来测试样本。在model(test)之前，
        需要加上model.eval()，否则的话，有输入数据，即使不训练，它也会改变权值。
        这是model中含有batch normalization层和Dropout所带来的的性质。
        :return:
        """
        self.actor.eval()
        self.actor_target.eval()
        self.critic.eval()
        self.critic_target.eval()

    def cuda(self):
        self.actor.cuda()
        self.actor_target.cuda()
        self.critic.cuda()
        self.critic_target.cuda()

    def observe(self, r_t, s_t1, done):
        if self.is_training:
            self.memory.append(self.s_t, self.a_t, r_t, done)
            self.s_t = s_t1

    def random_action(self):
        action = np.random.uniform(-1., 1., self.nb_actions)
        self.a_t = action
        return action

    def select_action(self, s_t, decay_epsilon=True):
        action = to_numpy(self.actor(to_tensor(np.array([s_t])))).squeeze(0)
        action += self.is_training * max(self.epsilon, 0) * self.random_process.sample()
        action = np.clip(action, -1., 1.)

        if decay_epsilon:
            self.epsilon -= self.depsilon
        self.a_t = action
        return action

    def reset(self, obs):
        self.s_t = obs
        self.random_process.reset_states()

    def load_weights(self, output):
        if output is None:
            return
        self.actor.load_state_dict(torch.load('{}/actor.pkl'.format(output)))
        self.critic.load_state_dict(torch.load('{}/critic.pkl'.format(output)))

    def save_model(self, output):
        torch.save(self.actor.state_dict(), '{}/actor.pkl'.format(output))
        torch.save(self.critic.state_dict(), '{}/critic.pkl'.format(output))

    def seed(self, s):
        torch.manual_seed(s)
        if USE_CUDA:
            torch.cuda.manual_seed(s)
