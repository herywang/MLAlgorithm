import gym
from maze_agent import AgentDSR
import gym_maze

if __name__ == '__main__':
    env = gym.make("maze-sample-5x5-v0")
    env.render()
    agent = AgentDSR(env, n_steps=40000, output_graph=True)
    agent.train()
