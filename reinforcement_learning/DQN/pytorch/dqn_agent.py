import gym
import gym_maze
import torch.nn as nn
import torch
from model import Net
import numpy as np
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import os

torch.manual_seed(0)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class Agent:
    def __init__(self, nb_states, nb_actions):

        args = {"hidden1": 128, "hidden2": 128}

        self.epsilon = 1.0
        self.epsilon_decay = 0.01
        self.epsilon_min = 0.07
        self.gamma = 0.9
        self.tau = 0.1

        self.memory = []
        self.memory_counter = 0
        self.memory_size = 1000
        self.batch_size = 128

        self.n_featuers = nb_states
        self.n_actions = nb_actions

        self.eval_net = Net(self.n_featuers, nb_actions, **args)
        self.target_net = Net(self.n_featuers, nb_actions, **args)
        self.opt = torch.optim.RMSprop(self.eval_net.parameters(), lr=0.001)
        # 确保两个神经网络初始权重一模一样
        self.target_net.load_state_dict(self.eval_net.state_dict())

    def learn(self):
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, self.batch_size, replace=False)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = np.array(self.memory, dtype=object)[sample_index]

        state = np.vstack(batch_memory[:, 0])
        state_ = np.vstack(batch_memory[:, 4])
        action = np.array(batch_memory[:, 1], dtype=np.int)
        reward = np.array(batch_memory[:, 2], dtype=np.float)
        done = np.array(batch_memory[:, 3], dtype=np.int)

        tensor_state = torch.tensor(state, dtype=torch.float32)
        tensor_state_ = torch.tensor(state_, dtype=torch.float32)

        state_action_value = self.eval_net(tensor_state)

        q_target = state_action_value.clone().detach()
        with torch.no_grad():
            # q_target = torch.zeros((self.batch_size, self.n_actions)).detach()
            next_state_action_value = self.target_net(tensor_state_)
            q_target[np.arange(0, self.batch_size), action] = torch.tensor(reward, dtype=torch.float32) + \
                                                              torch.squeeze(torch.tensor(1 - np.vstack(done), dtype=torch.float32)) * \
                                                              self.gamma * next_state_action_value.max(1).values

        loss = F.smooth_l1_loss(state_action_value, q_target)
        self.opt.zero_grad()
        loss.backward()
        for param in self.eval_net.parameters():
            # 防止梯度爆炸，截断梯度
            param.grad.data.clamp_(-1, 1)
        self.opt.step()
        return loss

    def store_transition(self, state, a, r, done, next_state):
        trainsition = (state, a, r, int(done), next_state)
        if self.memory_counter >= self.memory_size:
            self.memory[self.memory_counter % self.memory_size] = trainsition
        else:
            self.memory.append(trainsition)
        self.memory_counter += 1

    def choose_action(self, state, is_test=False):
        if is_test:
            self.epsilon = 1.0
        if self.epsilon > np.random.uniform():
            action = np.random.randint(0, self.n_actions)
        else:
            state = torch.Tensor(np.expand_dims(state, axis=0))
            with torch.no_grad():
                state_action_value = self.eval_net(state)
                action = int(torch.argmax(state_action_value, dim=1).data.numpy())
        return action

    def soft_update(self):
        for target_param, source_param in zip(self.target_net.parameters(), self.eval_net.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + source_param.data * self.tau)


if __name__ == '__main__':
    writer = SummaryWriter("./maze-log1")
    start_learn_after_n_steps = 100
    epoches = 3000
    step = 0
    env = gym.make("maze-sample-5x5-v0")
    # env = gym.make("CartPole-v0")
    agent = Agent(np.prod(env.observation_space.shape), env.action_space.n)
    for _ in range(epoches):
        observation = env.reset()
        done = False
        sum_reward = 0.
        while not done:
            env.render()
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            agent.store_transition(observation, action, reward, done, observation_)
            step += 1
            if step % 5 == 0:
                agent.learn()
            if step % 100 == 0:
                agent.soft_update()
            observation = observation_
            sum_reward += reward
        print("Epoches: %d， Sum Reward:%.3f, Epsilon: %.2f," % (_, sum_reward, agent.epsilon))
        writer.add_scalar('summary_reward', sum_reward, _)
        loss = agent.learn()
        writer.add_scalar("loss", loss.data.numpy(), _)
        agent.epsilon = (agent.epsilon - 0.01) if agent.epsilon > 0.07 else 0.0
        # if _ % 5 == 0:
        #     agent.soft_update()
    writer.close()
