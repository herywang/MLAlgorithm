import gym
import time
import threading
import keyboard
import gym_maze
import sys
import cv2

# env = gym.make('FetchReach-v1')
# env = gym.make('FetchPush-v1')
# env = gym.make('HalfCheetah-v2')
# env = gym.make('Humanoid-v2')
env = gym.make('maze-sample-10x10-v0')
# env = gym.make('CartPole-v0').unwrapped
# print("action_space:", env.action_space.n)
# print("action space low: high", env.action_space.low, env.action_space.high)
observation = env.reset()
# action = 0
# print(env.observation_space.shape)
# print(env.action_space.shape)
# def keyboard_listener(x):
#     global action
#     if x.event_type == 'up':
#         action = 0
#     elif x.name == 'left':
#         action = 3
#     elif x.name == 'right':
#         action = 2
#     elif x.name == 'down':
#         action = 1

def gym_env():
    global action, done
    # keyboard.hook(keyboard_listener)
    for _ in range(1000):
        done = False
        env.reset()
        while not done:
            env.render()
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            # print(observation)
            # break
    env.close()


if __name__ == '__main__':
    gym_env()

