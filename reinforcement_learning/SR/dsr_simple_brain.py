import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow import keras
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.losses import mean_squared_error
import numpy as np
from dsr_maze_env import DSRMaze


def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = mean_squared_error(y_true=targets, y_pred=model(inputs))
    return loss_value, tape.gradient(loss_value, model.trainable_variables)


class RL_Brain():
    def __init__(self, n_features, n_action, memory_size=10, batch_size=32, gamma=0.9, fi_size=10):
        self.n_features = n_features
        self.n_actions = n_action
        self.memory_size = memory_size
        self.replay_buffer = np.zeros((self.memory_size, n_features * 2 + 2), np.float)
        self.count = 0
        self.batch_size = batch_size
        self.gamma = gamma
        self.fi_size = fi_size
        self.opt = RMSprop()
        self.epsilon = 0.9

        self.encoder = keras.Sequential([
            Input((self.n_features,)),
            Dense(16, activation='relu', kernel_initializer='glorot_normal', name='encoder_1'),
            Dense(16, activation='relu', kernel_initializer='glorot_normal', name='encoder_2'),
            Dense(16, activation='relu', kernel_initializer='glorot_normal', name='encoder_3'),
            Dense(self.fi_size, activation='softmax', name='fi'),
        ])
        self.decoder = keras.Sequential([
            Input((self.fi_size,)),
            Dense(16, activation='relu', name='decoder_1', trainable=True),
            Dense(16, activation='relu', name='decoder_2', trainable=True),
            Dense(16, activation='relu', name='decoder_3', trainable=True),
            Dense(self.n_features, activation=None, name='decoder_output', trainable=True)
        ])
        self.branch_1_model = keras.Sequential([
            Input((self.fi_size,)),
            Dense(1, None, False, name='R')
        ])
        self.branch_2_model = [keras.Sequential([
            Input((self.fi_size,)),
            Dense(32, 'relu', name='action_%s/mu/layer1' % i),
            Dense(32, 'relu', name='action_%s/mu/layer2' % i),
            Dense(self.fi_size, 'softmax', name='action_%s/mu/layer3' % i)
        ]) for i in range(self.n_actions)]

    def learn(self, state, r, a, state_):
        # encoded = tf.one_hot(tf.argmax(self.encoder(state), axis=1), depth=self.fi_size)
        encoded = self.encoder(state)
        # encoded_ = tf.one_hot(tf.argmax(self.encoder(state_), axis=1), depth=self.fi_size)
        encoded_ = self.encoder(state_)
        decoded_state = self.decoder(encoded).numpy()
        q_value_ = self.calculate_q_value(state_)
        a_ = tf.argmax(q_value_).numpy()
        Ms_ = self.branch_2_model[a_](encoded_)

        loss1, grads1 = grad(self.decoder, encoded, decoded_state)
        loss2, grads2 = grad(self.branch_1_model, encoded, r)
        # tf.one_hot(tf.argmax(Ms_),depth=self.fi_size)
        loss3, grads3 = grad(self.branch_2_model[a], encoded, tf.nn.softmax(encoded + self.gamma * Ms_, axis=1))

        self.opt.apply_gradients(zip(grads1, self.decoder.trainable_variables))
        self.opt.apply_gradients(zip(grads2, self.branch_1_model.trainable_variables))
        self.opt.apply_gradients(zip(grads3, self.branch_2_model[a].trainable_variables))

    def choose_action(self, state, is_random=False):
        if is_random:
            return np.random.choice(self.n_actions)
        q_values = self.calculate_q_value(state)
        if len(set(q_values.numpy())) == 1 or np.random.uniform() < self.epsilon:
            action_index = np.random.choice(self.n_actions)
        else:
            action_index = tf.argmax(q_values, axis=0).numpy()
        return action_index

    def calculate_q_value(self, state):
        fi = self.encoder(state)
        encoded_fi = tf.one_hot(tf.argmax(fi, axis=1), depth=self.fi_size)
        mus = [self.branch_2_model[i](encoded_fi) for i in range(self.n_actions)]
        mus = tf.concat(tf.squeeze(mus), axis=1)
        w = self.branch_1_model.get_layer('R').get_weights()[0]
        q_value = tf.matmul(mus, w)
        return tf.squeeze(q_value)

    def append_to_replay_buffer(self, s, a, r, s_):
        transition = np.hstack([s, a, r, s_])
        self.replay_buffer[self.count % self.memory_size] = transition
        self.count += 1



if __name__ == '__main__':
    eps = 300
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
            brain.learn(state, reward, action_index, s_next)
            state = s_next
            # c += 1
            if done:
                env.reset()
