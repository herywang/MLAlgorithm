import numpy as np
import tensorflow._api.v2.compat.v1 as tf
from maze_env import DynaQMaze

tf.disable_v2_behavior()
tf.disable_eager_execution()


class DynaAgent:
    def __init__(self, exp_rate=0.3, lr=0.1, n_steps=5, episodes=1000, sess: tf.Session = None):
        self.maze = DynaQMaze()
        self.actions = self.maze.action_space
        self.n_actions = len(self.actions)
        self.state_actions = []  # state & action transition
        self.exp_rate = exp_rate
        self.lr = lr
        self.steps = n_steps
        self.episodes = episodes  # number of episodes going to play
        self.steps_per_episode = []
        self.state = self.maze.get_current_state()
        self.Q_values = {}
        # model function
        self.model = {}
        self.maze.render()
        if sess is None:
            self.sess = tf.Session()
        else:
            self.sess = sess
        self.writer1 = tf.summary.FileWriter('./log/r-1', self.sess.graph)
        self.writer2 = tf.summary.FileWriter('./log/r-2', self.sess.graph)
        self.tmp_tensor = tf.placeholder(tf.float32)
        self.all_reward_summary = tf.summary.scalar('all_reward', self.tmp_tensor)
        self.all_cnt_summary = tf.summary.scalar('all_cnt', self.tmp_tensor)

        self.write_op = tf.summary.merge_all()

    def choose_action(self):
        # epsilon-greedy
        action_index = 0
        mx_nxt_reward = -999
        if np.random.uniform(0, 1) <= self.exp_rate:
            action_index = np.random.choice(self.n_actions)
        else:
            # greedy action
            # if all actions have same value, then select randomly
            if self.get_key(self.state) not in self.Q_values.keys():
                self.Q_values[self.get_key(self.state)] = np.zeros(self.n_actions)
            if len(set(self.Q_values[self.get_key(self.state)])) == 1:
                action_index = np.random.choice(self.n_actions)
            else:
                for a_i in range(self.n_actions):
                    nxt_reward = self.Q_values[self.get_key(self.state)][a_i]
                    if nxt_reward >= mx_nxt_reward:
                        action_index = a_i
                        mx_nxt_reward = nxt_reward
        return action_index

    def reset(self):
        self.maze.reset()
        self.state = self.maze.get_current_state()
        self.state_actions = []

    def train(self):
        self.learn(True)
        self.Q_values = {}
        self.model = {}
        self.steps = 0
        self.learn(False)

    def learn(self, type=True):
        self.steps_per_episode = []
        for ep in range(self.episodes):
            cnt = 0.
            while not self.maze.end:
                action_index = self.choose_action()
                self.state_actions.append((self.state, action_index))
                nxtState, reward = self.maze.step(action_index)
                # 当前state的index.
                i_state = self.get_key(self.state)
                # index of next state.
                i_state_next = self.get_key(nxtState)
                # update Q-value
                if i_state_next not in self.Q_values.keys():
                    self.Q_values[i_state_next] = np.zeros(self.n_actions)
                if i_state not in self.Q_values.keys():
                    self.Q_values[i_state] = np.zeros(self.n_actions)
                self.Q_values[i_state][action_index] += self.lr * (reward + np.max(list(self.Q_values[i_state_next]))
                                                                   - self.Q_values[i_state][action_index])
                # update model
                if i_state not in self.model.keys():
                    self.model[i_state] = {}
                self.model[i_state][action_index] = (reward, nxtState)
                self.state = nxtState
                # loop n times to randomly update Q-value
                for _ in range(self.steps):
                    # randomly choose an state
                    rand_idx = np.random.choice(range(len(self.model.keys())))
                    _state = list(self.model)[rand_idx]
                    # randomly choose an action
                    rand_idx = np.random.choice(range(len(self.model[_state].keys())))
                    _action = list(self.model[_state])[rand_idx]
                    _reward, _nxtState = self.model[_state][_action]
                    self.Q_values[_state][_action] += self.lr * (_reward + np.max(list(self.Q_values[self.get_key(_nxtState)]))
                                                                 - self.Q_values[_state][_action])
                    # end of game
                cnt += 1.
            self.steps_per_episode.append(len(self.state_actions))
            self.reset()
            np_sum = np.sum(np.hstack(self.Q_values.values())) / cnt
            summary1 = self.sess.run(self.all_reward_summary, feed_dict={self.tmp_tensor: np_sum})
            summary2 = self.sess.run(self.all_cnt_summary, feed_dict={self.tmp_tensor: cnt})
            if type:
                self.writer1.add_summary(summary1, ep)
                self.writer1.add_summary(summary2, ep)
                self.writer1.flush()
            else:
                self.writer2.add_summary(summary1, ep)
                self.writer2.add_summary(summary2, ep)
                self.writer2.flush()

    def get_key(self, state):
        s = ''
        for v in state:
            s += str(v)
        return s


if __name__ == "__main__":
    N_EPISODES = 50
    # comparison
    sess = tf.Session()
    agent1 = DynaAgent(n_steps=5, episodes=100, sess=sess)
    # for i in range(N_EPISODES):
    agent1.train()
