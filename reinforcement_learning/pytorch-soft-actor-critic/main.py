import argparse

import gym
import numpy as np
import torch

from sac import SAC

parser = argparse.ArgumentParser(description="Pytorch Soft Actor Critic Args")
parser.add_argument('--env-name', default='HalfCheetah-v2')
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--tau', type=float, default=0.005, help='soft-update target 网络的系数')
parser.add_argument('--lr', type=float, default=0.0003)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--hidden_size', type=int, default=128)
parser.add_argument('--episodes', type=int, default=2000, help='训练轮数')
parser.add_argument('--start-learn-step', type=int, default=1000, help='随机探索多少轮之后开始学习')
parser.add_argument('--batch-size', type=int, default=128, help='训练批次大小')
parser.add_argument('--learn-after-steps', type=int, default=1, help='探索多少次学习一次')
parser.add_argument('--alpha', type=float, default=0.2, help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--automatic-entropy-tunning', type=bool, default=False, help='是否自动调整α')
parser.add_argument('--target-update-interval', type=int, default=10, help='学习多少次后开始更新target网络')
args = parser.parse_args()

if __name__ == '__main__':
    env = gym.make(args.env_name)
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    agent = SAC(env.observation_space.shape[0], env.action_space, args)

    total_steps = 0
    for i in range(args.episodes):
        episode_reward = 0.
        episode_steps = 0
        done = False
        state = env.reset()
        while not done:
            total_steps += 1
            if total_steps < args.start_learn_step:
                # 没有到开始学习的时候, 随机的选取action, 从环境中进行探索
                action = env.action_space.sample()
                next_state, reward, done, info = env.step(action)
                # 将一条交互数据存储到replay buffer中
                agent.store_transition(state, action, reward, next_state, done)
                state = next_state
            else:
                # evaluate=False: 采样的方式 获取action ,对应论文算法的 at ~ π(at | st)
                action = agent.select_action(state, evaluate=False)
                next_state, reward, done, info = env.step(action)
                agent.store_transition(state, action, reward, next_state, done)
                state = next_state
                if episode_steps % args.learn_after_steps == 0:
                    agent.learn()

            episode_reward += reward
            episode_steps += 1
            done = True if episode_steps == env._max_episode_steps else False
        print("episode {} summary reward is : {}".format(i, episode_reward))
