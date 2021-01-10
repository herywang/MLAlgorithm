from tensorflow.keras.layers import Dense, Input
from tensorflow import keras
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.losses import mean_squared_error
import tensorflow as tf

import numpy as np

from dsr_maze_env import DSRMaze


class RL_Brain():
    def __init__(self, n_features, n_action, memory_size=10, batch_size=32, gamma=0.9, fi_size=8):
        self.n_features = n_features
        self.n_actions = n_action
        self.memory_size = memory_size
        self.replay_buffer = np.zeros((self.memory_size, n_features * 2 + 2), np.float)
        self.count = 0
        self.batch_size = batch_size
        self.gamma = gamma

        self.opt = RMSprop()

        # 由于我们输入的状态向量纬度非常小（仅仅2纬度，因此没有必要再搞个auto-encoder对state进行编码解码），我们直接将state值进行输入。
        self.input_states = Input((self.n_features,), name='input_states')

        self.branch_1_model = keras.Sequential([
            Input((2,)),
            Dense(1, None, False, name='R')
        ])

        self.branch_2_model = [keras.Sequential([
            Input((2,)),
            Dense(5, 'relu', name='mu/m%s/layer1' % i),
            Dense(5, 'relu', name='mu/m%s/layer2' % i),
            Dense(2, name='mu/m%s/layer3' % i)
        ], name='branch_%s' % i) for i in range(self.n_actions)]

    def learn_w(self, state, r):
        with tf.GradientTape() as tape:
            pred = self.branch_1_model(state)
            loss = mean_squared_error(r, pred)
        grads = tape.gradient(loss, self.branch_1_model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.branch_1_model.trainable_variables))

    def learn_mu(self, state, state_, action_index):
        w = self.branch_1_model.get_layer('R').get_weights()[0]
        mus_ = []
        for i in range(self.n_actions):
            mus_.append(self.branch_2_model[i](state_))
        mus_ = np.squeeze(mus_)
        max_index = np.argmax(np.squeeze(np.matmul(mus_, w)), axis=0)
        with tf.GradientTape() as tape:
            pred = self.branch_2_model[action_index](state)
            label = state + self.gamma * mus_[max_index]
            loss = mean_squared_error(label, pred)
        grads = tape.gradient(loss, self.branch_2_model[action_index].trainable_variables)
        self.opt.apply_gradients(zip(grads, self.branch_2_model[action_index].trainable_variables))
        self.branch_2_model[action_index] = self.branch_2_model[action_index]

    def choose_action(self, state, is_random=False):
        if is_random:
            return np.random.choice(self.n_actions)
        w = self.branch_1_model.get_layer('R').get_weights()[0]
        mus = []
        for i in range(self.n_actions):
            pred = self.branch_2_model[i](state)
            mus.append(pred)
        mus = np.squeeze(mus)
        rs = np.squeeze(np.matmul(mus, w))
        if len(set(rs)) == 1:
            action_index = np.random.choice(self.n_actions)
        else:
            action_index = np.argmax(rs)
        return action_index

    def append_to_replay_buffer(self, s, a, r, s_):
        transition = np.hstack([s, a, r, s_])
        self.replay_buffer[self.count % self.memory_size] = transition
        self.count += 1


if __name__ == '__main__':
    eps = 100
    env = DSRMaze('dsr-maze')
    brain = RL_Brain(2, 4, memory_size=10000)

    c = 0
    for i in range(eps):
        state = env.get_current_state()
        done = False

        while not done:
            action_index = brain.choose_action(state)
            s_next, reward, done = env.step(action_index)
            brain.append_to_replay_buffer(np.squeeze(state), action_index, reward, np.squeeze(s_next))
            brain.learn_w(state, reward)
            # 训练第二部分：m_alpha
            brain.learn_mu(state, s_next, action_index)
            state = s_next
            # c += 1
            if done:
                env.reset()
