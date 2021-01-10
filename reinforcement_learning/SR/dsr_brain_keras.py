'''
tensorflow: 2.3.0
'''
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow import keras
from dsr_maze_env import DSRMaze
import numpy as np


class RL_Brain():
    def __init__(self, n_features, n_action, memory_size=10, batch_size=32, gamma=0.9, fi_size=8):
        self.n_features = n_features
        self.n_actions = n_action
        self.whole_model = self.build_model()
        self.memory_size = memory_size
        self.replay_buffer = np.zeros((self.memory_size, n_features * 2 + 2), np.float)
        self.count = 0
        self.batch_size = batch_size
        self.gamma = gamma
        self.fi_size = fi_size

        self.input_states = Input((self.n_features,), name='input_states')
        self.input_fi = Input((self.fi_size,), name='input_fi')

        self.branch_1_model = self._build_branch1(self.input_states)
        self.branch_2_model = self._build_branch2(self.input_fi)

    def _build_branch1(self, input_):
        encode_layer1 = Dense(10, 'relu', name='encode/layer1')(input_)
        encode_layer2 = Dense(10, 'relu', name='encode/layer2')(encode_layer1)
        fi = Dense(self.fi_size, 'relu', name='fi')(encode_layer2)

        decode_layer1 = Dense(10, 'relu', name='decode/layer1')(fi)
        decode_layer2 = Dense(10, 'relu', name='decode/layer2')(decode_layer1)
        decode_layer3 = Dense(2, name='decode/layer3')(decode_layer2)

        R = Dense(1, None, False, name='R')(fi)
        model = Model(inputs=input_, outputs=[fi, decode_layer3, R])

        loss = {'R': 'mse', 'decode/layer3': 'mse'}
        model.compile('adam', loss, metrics=['mse'])
        return model

    def _build_branch2(self, input_):
        mus = []
        loss = {}
        for i in range(self.n_actions):
            mu = Dense(5, 'relu', name='mu/m%s/layer1' % i, kernel_initializer='zero')(input_)
            mu = Dense(5, 'relu', name='mu/m%s/layer2' % i, kernel_initializer='zero')(mu)
            mu = Dense(8, 'relu', name='mu/m%s/layer3' % i, kernel_initializer='zero')(mu)
            mus.append(mu)
            loss['mu/m%s/layer3' % i] = 'mae'

        model = Model(inputs=input_, outputs=mus)
        model.compile(loss=loss, metrics=['mse'])
        return model

    def build_model(self):
        input_ = Input(shape=(self.n_features,), name='input')
        layer1 = Dense(64, 'relu', name='encode/layer1')(input_)
        layer2 = Dense(64, 'relu', name='encode/layer2')(layer1)
        layer3 = Dense(10, 'relu', name='encode/layer3')(layer2)
        fai = Dense(5, 'relu', name='fai')(layer3)
        decoder1 = Dense(10, 'relu', name='decode/layer1')(fai)
        decoder2 = Dense(64, 'relu', name='decode/layer2')(decoder1)
        decoder3 = Dense(64, 'relu', name='decode/layer3')(decoder2)
        s_hat = Dense(self.n_features, name='output_s_hat')(decoder3)
        R = Dense(1, name='R', use_bias=False)(fai)
        mus = []
        for i in range(self.n_actions):
            mu = Dense(10, 'relu', name='mu/m%s/layer1' % i, kernel_initializer='zero')(fai)
            mu = Dense(28, 'relu', name='mu/m%s/layer2' % i, kernel_initializer='zero')(mu)
            mu = Dense(5, 'relu', name='mu/m%s/layer3' % i, kernel_initializer='zero')(mu)
            mus.append(mu)
        outputs = [fai, R, s_hat]
        outputs = outputs + mus
        model = Model(inputs=input_, outputs=outputs)
        loss = {'R': 'mse', 'output_s_hat': 'mse'}
        for i in range(self.n_actions):
            loss['mu/m%s/layer3' % i] = 'mse'
        model.compile(keras.optimizers.RMSprop(), loss=loss, metrics=['mse'])
        return model

    def learn_theta_and_w(self, states, r):
        self.branch_1_model.fit({'input_states': states}, {'decode/layer3': states, 'R': r},batch_size=100, epochs=10, verbose=0)
        self.branch_1_model.save('./model/branch_1_model.h5')

    def learn_mu(self, state, state_, action_index):
        fi = self.branch_1_model.predict(state)[0]
        fi_ = self.branch_1_model.predict(state_)[0]

        w = self.branch_1_model.get_layer('R').get_weights()[0]
        mus_ = np.squeeze(self.branch_2_model.predict(fi_))
        max_index = np.argmax(np.squeeze(np.matmul(mus_, w)), axis=0)

        label = fi + self.gamma * mus_[max_index]
        outputs = {}
        for i in range(self.n_actions):
            outputs['mu/m%s/layer3' % i] = np.zeros((1, self.fi_size))
        outputs['mu/m%s/layer3' % action_index] = label
        self.branch_2_model.fit({'fi': fi}, outputs, epochs=10, verbose=1)

    def choose_action(self, state, is_random=False):
        if is_random:
            return np.random.choice(self.n_actions)
        w = self.branch_1_model.get_layer('R').get_weights()[0]
        fi = self.branch_1_model.predict(state)[0]
        mus = np.squeeze(self.branch_2_model.predict(fi))
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

    @tf.function
    def traceme(self, input_1, input_2):
        return self.whole_model(input_1), self.branch_1_model(input_1), self.branch_2_model(input_2)

    def visualize_model(self):
        log_dir = './log'
        writer = tf.summary.create_file_writer(log_dir)
        tf.summary.trace_on(graph=True)
        self.traceme(tf.zeros(shape=(5, 2)), tf.zeros(shape=(1, self.fi_size)))
        with writer.as_default():
            tf.summary.trace_export('model_trace', step=0)
        # keras.utils.plot_model(self.whole_model, './dsr_model.png', True, True)


def load_trained_model():
    model: Model = load_model('./model/branch_1_model.h5')
    print(model.predict(np.array([[-0.875, -0.125]])))


if __name__ == '__main__':
    eps = 100
    env = DSRMaze('dsr-maze')
    brain = RL_Brain(2, 4, memory_size=10000)

    for i in range(eps):
        state = env.get_current_state()
        done = False
        c = 0
        while not done:
            action_index = brain.choose_action(state)
            s_next, reward, done = env.step(action_index)
            brain.append_to_replay_buffer(np.squeeze(state), action_index, reward, np.squeeze(s_next))
            choices = np.random.choice(brain.count if brain.count < brain.memory_size else brain.memory_size, 100, replace=True)
            states = brain.replay_buffer[choices, :brain.n_features]
            r = brain.replay_buffer[choices, brain.n_features + 1]
            if brain.count > 100:
                # 首先训练第一部分： 编码器和
                if c<20:
                    brain.learn_theta_and_w(states, r)
                    c+=1
                # 训练第二部分：m_alpha
                brain.learn_mu(state, s_next, action_index)
                state = s_next
            if done:
                env.reset()
# print(brain.model.get_layer('R').get_weights())
