'''
tensorflow: 2.3.0
'''
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import RMSprop, Adam
from dsr_maze_env import DSRMaze
import numpy as np
import matplotlib.pyplot as plt


class RL_Brain():
    def __init__(self, n_features, n_action, memory_size=10, batch_size=32, gamma=0.9, phi_size=15):
        self.n_features = n_features
        self.n_actions = n_action
        self.memory_size = memory_size
        self.replay_buffer = np.zeros((self.memory_size, n_features * 2 + 2), np.float)
        self.count = 0
        self.batch_size = batch_size
        self.gamma = gamma
        self.phi_size = phi_size
        self.epsilon = 0.9  # 默认有0.1的随机度
        self.autoencoder_model, self.mus_model, self.reward_model = self.build_model()

        self.autoencoder_opt = Adam()
        self.reward_opt = Adam()
        self.mus_opt = Adam()

    def build_model(self):
        input_state = Input(shape=(self.n_features,), name='input')
        input_phi = Input(shape=(self.phi_size,), name='input_phi')

        layer1 = Dense(32, 'relu', name='encode/layer1')(input_state)
        layer2 = Dense(32, 'relu', name='encode/layer2')(layer1)
        layer3 = Dense(10, 'relu', name='encode/layer3')(layer2)
        phi = Dense(15, 'relu', name='phi')(layer3)
        decoder1 = Dense(10, 'relu', name='decode/layer1')(phi)
        decoder2 = Dense(32, 'relu', name='decode/layer2')(decoder1)
        decoder3 = Dense(32, 'relu', name='decode/layer3')(decoder2)
        decoded = Dense(self.n_features, name='output_s_hat')(decoder3)

        # reward_layer1 = Dense(16, name='R_1')(input_phi)
        # reward_layer2 = Dense(16, name='R_2')(reward_layer1)
        R = Dense(1, name='R')(input_phi)
        mus = []
        for i in range(self.n_actions):
            mu = Dense(10, 'relu', name='mu/m%s/layer1' % i)(input_phi)
            mu = Dense(10, 'relu', name='mu/m%s/layer2' % i)(mu)
            mu = Dense(15, 'relu', name='mu/m%s/layer3' % i)(mu)
            m = Model(inputs=input_phi, outputs=mu)
            mus.append(m)

        outputs = [phi, decoded]
        model = Model(inputs=input_state, outputs=outputs)
        reward_model = Model(inputs=input_phi, outputs=R)
        return model, mus, reward_model

    def learn(self):
        # choices = np.random.choice(self.count if self.count < self.memory_size else self.memory_size, self.batch_size, replace=True)
        states = np.expand_dims(self.replay_buffer[(self.count-1) % self.memory_size, :self.n_features], 0)
        states_ = np.expand_dims(self.replay_buffer[(self.count-1) % self.memory_size, -self.n_features:], 0)
        r = np.expand_dims(self.replay_buffer[(self.count-1) % self.memory_size, self.n_features + 1], 0)
        a = int(self.replay_buffer[(self.count-1) % self.memory_size, self.n_features])
        o_phi_t, o_s_hat = self.autoencoder_model(states)  # 模型输出的phi, reward, s_hat
        # Training auto-encoder model.

        with tf.GradientTape() as tape:
            loss1 = tf.reduce_mean(tf.square(states - self.autoencoder_model(states)[1]))
        if loss1.numpy() > 1e-7:
            self.autoencoder_opt.minimize(loss1, self.autoencoder_model.trainable_variables, tape=tape)

        # Training reward model.
        # for i in range(2):
        o_r = self.reward_model(o_phi_t)
        if r[0] == 1.:
            for i in range(10):
                with tf.GradientTape() as tape:
                    loss2 = tf.reduce_mean(tf.square(r - self.reward_model(o_phi_t)))
                self.reward_opt.minimize(loss2, self.reward_model.trainable_variables, tape=tape)
        else:
            with tf.GradientTape() as tape:
                loss2 = tf.reduce_mean(tf.square(r - self.reward_model(o_phi_t)))
            self.reward_opt.minimize(loss2, self.reward_model.trainable_variables, tape=tape)
        o_phi_t_, _ = self.autoencoder_model(states_)
        mus_ = tf.squeeze(tf.stack([self.mus_model[i](o_phi_t_) for i in range(self.n_actions)]))
        q = self.reward_model(mus_)
        max_q_action_index = tf.argmax(tf.squeeze(q)).numpy()
        # Training M loss
        # for i in range(2):
        with tf.GradientTape() as tape:
            loss3 = tf.reduce_mean(tf.square(o_phi_t + self.gamma * mus_[max_q_action_index] - self.mus_model[a](o_phi_t)))
        self.mus_opt.minimize(loss3, self.mus_model[a].trainable_variables, tape=tape)
        return o_r

    def choose_action(self, state, is_random=False):
        if is_random or np.random.uniform() > self.epsilon:
            return np.random.choice(self.n_actions)
        phi = self.autoencoder_model(state)[0]
        mus = tf.squeeze(tf.stack([self.mus_model[i](phi) for i in range(self.n_actions)]))
        q = self.reward_model(mus)
        if len(set(np.squeeze(q.numpy()))) == 1:
            action_index = np.random.choice(self.n_actions)
        else:
            action_index = tf.argmax(q).numpy()[0]
        return action_index

    def append_to_replay_buffer(self, s, a, r, s_):
        transition = np.hstack([s, a, r, s_])
        self.replay_buffer[self.count % self.memory_size] = transition
        self.count += 1

    @tf.function
    def traceme(self, input_1, input_2):
        return self.autoencoder_model(input_1)

    def visualize_model(self):
        log_dir = './log'
        writer = tf.summary.create_file_writer(log_dir)
        tf.summary.trace_on(graph=True)
        self.traceme(tf.zeros(shape=(5, 2)), tf.zeros(shape=(1, self.phi_size)))
        with writer.as_default():
            tf.summary.trace_export('model_trace', step=0)
        # keras.utils.plot_model(self.whole_model, './dsr_model.png', True, True)


def load_trained_model():
    model: Model = load_model('./model/branch_1_model.h5')
    print(model.predict(np.array([[-0.875, -0.125]])))


if __name__ == '__main__':
    eps = 100
    env = DSRMaze('dsr-maze')
    brain = RL_Brain(2, 4, memory_size=100, phi_size=15)
    rewards = []
    for i in range(eps):
        state = env.get_current_state()
        done = False
        c = 0
        brain.epsilon = 0.1 * i if 0.1 * i < 1. else 0.9
        sum_reward = 0.
        history_rewards = 0.
        while not done:
            action_index = brain.choose_action(state)
            s_next, reward, done = env.step(action_index)
            sum_reward = sum_reward + reward - 0.01
            brain.append_to_replay_buffer(np.squeeze(state), action_index, reward, np.squeeze(s_next))
            state = s_next
            o_r = brain.learn().numpy()[0,0]
            if done:
                env.reset()
        history_rewards = history_rewards * 0.9 + sum_reward * 0.1
        rewards.append(history_rewards)
    plt.figure()
    plt.plot(rewards)
    plt.show()
