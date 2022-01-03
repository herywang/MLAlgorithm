from model import Net
from torch.optim import Adam
import torch
import torch.nn as nn
import argparse
from torch.nn import Module
import numpy as np
import gym
from environment import MazeEnvironmentWrapper
from environment import DSRMaze
import gym_maze
import os
import copy

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class HERAgent:
    def __init__(self, env, nb_states: int, nb_actions: int, args):
        self.env = env

        if args.seed > 0:
            self.seed(args.seed)
        network_cofig = {"hidden1": args.hidden1, "hidden2": args.hidden2}

        self.use_double_dqn = args.use_double_dqn

        self.nb_states = nb_states
        self.nb_actions = nb_actions

        self.gamma = args.gamma
        self.episode = args.episode
        self.explore_steps = args.explore_steps
        self.epsilon = args.epsilon
        self.epsilon_decay = 0.05
        self.epsilon_min = 0.07

        self.eval_network = Net(nb_states, nb_actions, **network_cofig)
        self.target_network = Net(nb_states, nb_actions, **network_cofig)
        # 关闭target_network训练
        self.target_network.eval()

        self.eval_network_opt = Adam(self.eval_network.parameters(), lr=args.rate)

        self.loss_function = nn.SmoothL1Loss()

        # 确保刚开始创建的两个网络参数值一模一样。
        self.hard_update(self.target_network, self.eval_network)

        self.memory_size = 1000
        self.memory = []
        self.memory_counter = 0
        self.batch_size = 128

        self.optimization_steps = 10  # 每一轮学习执行多少步优化
        self.K = 4  # the number of random future states.

    def choose_action(self, state, goal, is_random=False, is_test=False):
        if is_test:
            self.epsilon = 0.01
        if is_random:
            action = np.random.randint(0, 4)
        else:
            if np.random.uniform() > self.epsilon:
                x = torch.Tensor(np.expand_dims(np.concatenate((state, goal)), axis=0))
                action = int(np.argmax(self.eval_network(x).data.numpy()))
            else:
                action = np.random.randint(0, 4)
        return action

    def store_transition(self, s, a, r, s_, done, goal):
        self.memory.append((s, s_, goal, a, r, int(done)))
        self.memory_counter += 1
        if self.memory_counter > self.memory_size:
            self.memory = self.memory[-self.memory_size:]

    def learn(self):
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, self.batch_size, replace=False)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = np.array(self.memory)[sample_index]

        states = np.stack(batch_memory[:, 0])
        states_ = np.stack(batch_memory[:, 1])
        goal = np.stack(batch_memory[:, 2])
        a = np.stack(batch_memory[:, 3])
        r = np.stack(batch_memory[:, 4])
        done = np.stack(batch_memory[:, 5])

        with torch.no_grad():
            q_next = self.target_network(torch.Tensor(np.concatenate([states_, goal], axis=1)))
            if self.use_double_dqn:
                q_eval_next = self.eval_network(torch.Tensor(np.concatenate([states_, goal], axis=1)))
                max_act4next = torch.argmax(q_eval_next, 1)
                selected_q_next = q_eval_next[np.arange(0, self.batch_size), max_act4next]
            else:
                selected_q_next = torch.max(q_next, 1)
            q_target = q_next.clone().detach()
            q_target[np.arange(0, self.batch_size), a] = torch.Tensor(r) + torch.Tensor(self.gamma * (1 - done)) * selected_q_next

        q_target_value = torch.clip(q_target, -1. / (1 - self.gamma), 1. / (1 - self.gamma))

        self.eval_network.zero_grad()
        q_batch = self.eval_network(torch.Tensor(np.concatenate([states, goal], axis=1)))
        value_loss = self.loss_function(q_batch, q_target_value)
        value_loss.backward()
        self.eval_network_opt.step()
        return value_loss

    def train(self):
        step = 0
        for j in range(self.episode):
            observatioin, real_goal = self.env.reset()
            done = False
            episode_reward = 0.
            episode_step = 0
            while not done:
                self.env.render()
                if step < self.explore_steps:
                    action = self.choose_action(observatioin, real_goal, True)
                else:
                    action = self.choose_action(observatioin, real_goal)
                observatioin_, reward, done, _ = self.env.step(action)
                self.store_transition(observatioin, action, reward, observatioin_, done, real_goal)
                step += 1
                episode_step += 1
                episode_reward += reward
                observatioin = observatioin_
                if step % 5 == 0:
                    self.learn()
            episode_avg_reward = episode_reward / episode_step

            # HER算法核心：
            length_ = self.memory_size if self.memory_counter > self.memory_size else self.memory_counter
            for t in range(length_):
                for k in range(self.K):
                    future = np.random.randint(t, length_)
                    goal = self.memory[future][1]
                    state = self.memory[t][0]
                    state_ = self.memory[t][1]
                    action = self.memory[t][3]
                    done = np.array_equal(state_, goal)
                    reward = 1 if done else 0
                    self.store_transition(state, action, reward, state_, done, goal)

            # 探索结束，开始学习
            if step >= self.explore_steps:
                loss = 0.
                for i in range(self.optimization_steps):
                    loss += self.learn()
                # 训练40轮，更新target_network参数
                self.soft_update(self.target_network, self.eval_network, 0.001)
                # self.hard_update(self.target_network, self.eval_network)
                loss /= self.optimization_steps
                print("Episode: %d ,epsilon: %.2f,  Step: %d, episode_avg_reward: %.5f, Loss: %.5f" % (j, self.epsilon, step, episode_avg_reward, loss))
                self.epsilon = (self.epsilon - self.epsilon_decay) if self.epsilon > self.epsilon_min else self.epsilon_min

    def seed(self, seed):
        torch.manual_seed(seed)

    def hard_update(self, target: Module, source: Module):
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(source_param.data)

    def soft_update(self, target: Module, source: Module, tau):
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + source_param.data * tau)


def run():
    parser = argparse.ArgumentParser(description="HER algorithm neural network config parameters.")
    parser.add_argument("--hidden1", default=128, type=int, help="The number of units of nerual network hidden layer1")
    parser.add_argument("--hidden2", default=128, type=int, help="The number of units of nerual network hidden layer2")
    parser.add_argument("--init_w", default=0.001, type=float, help="")
    parser.add_argument("--rate", default=0.001, type=float, help="Learning rate")
    parser.add_argument("--use_double_dqn", default=True, type=bool, help="Wherther to use double DQN | 是否使用double Q learning")
    parser.add_argument("--gamma", default=0.95, type=float, help="discount coefficients | 折扣系书")
    parser.add_argument("--episode", default=10000, type=int, help="Training Episode. | 总共训练几轮")
    parser.add_argument("--epsilon", default=0.999, type=float, help="Epsilon coefficent | 动作选择的随机度")
    parser.add_argument("--explore_steps", default=500, type=int, help="Starting learning after some steps.| 探索多少轮之后再开始学习")
    parser.add_argument("--seed", default=0, type=int, help="random seed. | 随机数种子")

    args = parser.parse_args()

    # env = MazeEnvironmentWrapper(gym.make('maze-sample-5x5-v0'))
    env = DSRMaze()
    # env = BitFlip(10, 'sparse')
    # nb_features = np.prod(np.array(env.observation_space.shape))
    # nb_features = 10
    her_agent = HERAgent(env, 2, 4, args)
    her_agent.train()


if __name__ == '__main__':
    run()
