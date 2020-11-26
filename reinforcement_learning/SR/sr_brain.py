import numpy as np
import matplotlib.pyplot as plt
import utils
from sklearn.decomposition import PCA
from gridworld import SimpleGrid
import tensorflow as tf

grid_size = 7
pattern="four_rooms"
env = SimpleGrid(grid_size, block_pattern=pattern, obs_mode="index")
env.reset(agent_pos=[0,0], goal_pos=[0, grid_size-1])
# plt.imshow(env.grid)
# plt.show()
sess = tf.Session()
# sess.run(tf.global_variables_initializer())

class TabularSuccessorAgent(object):
    def __init__(self, n_state, n_action, learning_rate, gamma):
        self.n_state = n_state
        self.n_action = n_action
        # shape: (state_size, action_size)
        self.M = np.stack([np.identity(n_state) for i in range(n_action)])
        self.w = np.zeros([n_state])
        self.learning_rate = learning_rate
        self.gamma = gamma
        
    def Q_estimates(self, state, goal=None):
        # Generate Q values for all actions.
        if goal == None:
            goal = self.w
        else:
            goal = utils.onehot(goal, self.n_state)
        return np.matmul(self.M[:,state,:],goal)
    
    def choose_action(self, state, goal=None, epsilon=0.0):
        # Samples action using epsilon-greedy approach
        if np.random.uniform(0, 1) < epsilon:
            action = np.random.randint(self.n_action)
        else:
            Qs = self.Q_estimates(state, goal)
            action = np.argmax(Qs)
        return action
    
    def update_w(self, current_exp):
        # A simple update rule. current_exp's shape: [state, a, next_state, r, done]
        s_ = current_exp[2]
        r = current_exp[3]
        error = r - self.w[s_]
        self.w[s_] += self.learning_rate * error        # w相当于公式中对R(s)的更新
        return error
    
    def update_sr(self, current_exp, next_exp):
        # SARSA TD learning rule
        # update the M(s, s', a)
        s = current_exp[0]      # current state
        s_a = current_exp[1]    # choosed action
        s_ = current_exp[2]    # next state
        s_a_1 = next_exp[1]     # next state choosed action
        r = current_exp[3]      # reward in current state
        d = current_exp[4]      # wheather the current state is terminal
        I = utils.onehot(s, env.state_size)    # transform current state to one-hot vector 
        if d:            
            td_error = (I + self.gamma * utils.onehot(s_, env.state_size) - self.M[s_a, s, :]) 
        else:
            td_error = (I + self.gamma * self.M[s_a_1, s_, :] - self.M[s_a, s, :])
        self.M[s_a, s, :] += self.learning_rate * td_error
        return td_error

train_episode_length = 50
test_episode_length = 50
episodes = 2000
gamma = 0.95
lr = 5e-2
train_epsilon = 1.0
test_epsilon = 0.1

agent = TabularSuccessorAgent(env.state_size, env.action_size, lr, gamma)

experiences = []            #replay buffer [(current_state), action, (next_state), done]  == experience.shape
test_experiences = []
test_lengths = []
lifetime_td_errors = []

for i in range(episodes):
    # Train phase
    agent_start = [0,0]
    if i < episodes // 2:
        goal_pos = [0, grid_size-1]
    else:
        if i == episodes // 2:
            print("\nSwitched reward locations")
        goal_pos = [grid_size-1,grid_size-1]
    env.reset(agent_pos=agent_start, goal_pos=goal_pos)
    state = env.observation
    episodic_error = []
    for j in range(train_episode_length):
        action = agent.choose_action(state, epsilon=train_epsilon)
        reward = env.step(action)
        state_next = env.observation
        done = env.done
        experiences.append([state, action, state_next, reward, done])
        state = state_next
        if (j > 1):
            # 至少会和环境交互两次. 当前experience: experience[-1], 前一次experience[-2]
            td_sr = agent.update_sr(experiences[-2], experiences[-1])
            td_w = agent.update_w(experiences[-1])
            episodic_error.append(np.mean(np.abs(td_sr)))
        if env.done:
            td_sr = agent.update_sr(experiences[-1], experiences[-1])
            episodic_error.append(np.mean(np.abs(td_sr)))
            break
    lifetime_td_errors.append(np.mean(episodic_error))
    
    # Test phase
    env.reset(agent_pos=agent_start, goal_pos=goal_pos)
    state = env.observation
    for j in range(test_episode_length):
        action = agent.choose_action(state, epsilon=test_epsilon)
        reward = env.step(action)
        state_next = env.observation
        test_experiences.append([state, action, state_next, reward])
        state = state_next
        if env.done:
            break
    test_lengths.append(j)
    
    if i % 50 == 0:
        print('\rEpisode {}/{}, TD Error: {}, Test Lengths: {}'
              .format(i, episodes, np.mean(lifetime_td_errors[-50:]), 
                      np.mean(test_lengths[-50:])), end='')

fig = plt.figure(figsize=(10, 6))

ax = fig.add_subplot(2, 2, 1)
ax.plot(lifetime_td_errors)
ax.set_title("TD Error")
ax = fig.add_subplot(2, 2, 2)
ax.plot(test_lengths)
ax.set_title("Episode Lengths")
plt.show()
