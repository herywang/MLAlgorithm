import tensorflow as tf
import numpy as np
import gym
from gym.spaces import Box
from OUActorNoise import OUActionNoise

tf.disable_v2_behavior()
tf.disable_eager_execution()

env = gym.make('Pendulum-v0')
env.seed(1)  # reproducible


class Agent:
    def __init__(self, env):
        self.env = env

        assert isinstance(self.env.observation_space, Box), "observation space must be continuous."
        assert isinstance(self.env.action_space, Box), "acton space must be continuous."
        self.feature_shape = self.env.observation_space.shape
        self.n_features = np.prod(np.array(self.feature_shape))  # 内积, 获取到特征数量
        self.n_actions = self.env.action_space.shape[0]

        self.upper_bound = self.env.action_space.high[0]  # action_space 数值上限
        self.lower_bound = self.env.action_space.low[0]  # action_space 数值下限

        # 序贯噪音
        self.ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(0.2) * np.ones(1))

        self.memory_size = 256
        # self.memory = Memory(self.memory_size, self.n_features * 2 + self.n_actions + 2)
        self.memory = np.zeros((self.memory_size, self.n_features * 2 + self.n_actions + 2))
        self.memory_counter = 0
        self.batch_size = 32

        self.gamma = 0.9
        self.epsilon = 1.0
        self.epsilon_min = 0.07
        self.epsilon_decrement = (self.epsilon - self.epsilon_min) / 8000.

        self.p_s = tf.placeholder(tf.float32, [None, self.n_features], name='state')
        self.p_s_ = tf.placeholder(tf.float32, [None, self.n_features], name='state_')
        self.p_a = tf.placeholder(tf.float32, [None, self.n_actions], name='action')
        self.p_a_ = tf.placeholder(tf.float32, [None, self.n_actions], name='action_')
        self.p_r = tf.placeholder(tf.float32, [None, self.n_actions], name='reward')
        self.p_target_critic = tf.placeholder(tf.float32, [None, 1], name='target_critic')
        self.p_q_gradient = tf.placeholder(tf.float32, [None, self.n_actions], name='q_gradient')

        self.sess = tf.Session()

        self.build_model()

        critic_eval_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'critic_eval_net')
        critic_target_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'critic_target_net')

        actor_eval_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'actor_eval_net')
        actor_target_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'actor_target_net')

        self.critic_assign_op = [tf.assign(t, e) for t, e in zip(critic_target_params, critic_eval_params)]
        self.actor_assign_op = [tf.assign(t, e) for t, e in zip(actor_target_params, actor_eval_params)]

        self.sess.run(tf.global_variables_initializer())

    def update_target(self):
        self.sess.run([self.critic_assign_op, self.actor_assign_op])

    def build_model(self):
        # ------------------ initializers ------------------ #
        from tensorflow.python.ops.init_ops import VarianceScaling

        def lecun_normal(seed=None):
            return VarianceScaling(scale=1., mode='fan_in', distribution='truncated_normal', seed=seed)

        w_initializer = lecun_normal()

        # ------------------ build critic evaluate_net ------------------ #
        with tf.variable_scope('critic_eval_net'):
            s_dense1 = tf.layers.dense(self.p_s, units=128, activation=tf.nn.relu, kernel_initializer=w_initializer, name='s_dense1')
            s_dense1 = tf.layers.batch_normalization(s_dense1, name='s_dense1_bn')
            s_dense2 = tf.layers.dense(s_dense1, units=128, activation=tf.nn.relu, kernel_initializer=w_initializer, name='s_dense2')
            s_dense2 = tf.layers.batch_normalization(s_dense2, name='s_dense2_bn')

            a_dense1 = tf.layers.dense(self.p_a, units=128, activation=tf.nn.relu, kernel_initializer=w_initializer, name='a_dense1')
            a_dense1 = tf.layers.batch_normalization(a_dense1, name='a_dense1_bn')
            a_dense2 = tf.layers.dense(a_dense1, units=128, activation=tf.nn.relu, kernel_initializer=w_initializer, name='a_dense2')
            a_dense2 = tf.layers.batch_normalization(a_dense2, name='a_dense2_bn')

            dense1 = tf.concat([s_dense2, a_dense2], axis=1, name='concat')
            dense1 = tf.layers.dense(dense1, units=64, activation=tf.nn.relu, kernel_initializer=w_initializer, name='dense1')
            dense1 = tf.layers.batch_normalization(dense1, name='bn2')
            dense2 = tf.layers.dense(dense1, units=64, activation=tf.nn.relu, kernel_initializer=w_initializer, name='dense2')
            self.critic_eval = tf.layers.dense(dense2, 1, kernel_initializer=w_initializer, name='critic_eval')

        # ------------------ build critic target net -------------------- #
        with tf.variable_scope('critic_target_net'):
            s_dense1 = tf.layers.dense(self.p_s_, units=128, activation=tf.nn.relu, kernel_initializer=w_initializer, name='s_dense1')
            s_dense1 = tf.layers.batch_normalization(s_dense1, name='s_dense1_bn')
            s_dense2 = tf.layers.dense(s_dense1, units=128, activation=tf.nn.relu, kernel_initializer=w_initializer, name='s_dense2')
            s_dense2 = tf.layers.batch_normalization(s_dense2, name='s_dense2_bn')

            a_dense1 = tf.layers.dense(self.p_a_, units=128, activation=tf.nn.relu, kernel_initializer=w_initializer, name='a_dense1')
            a_dense1 = tf.layers.batch_normalization(a_dense1, name='a_dense1_bn')
            a_dense2 = tf.layers.dense(a_dense1, units=128, activation=tf.nn.relu, kernel_initializer=w_initializer, name='a_dense2')
            a_dense2 = tf.layers.batch_normalization(a_dense2, name='a_dense2_bn')

            dense1 = tf.concat([s_dense2, a_dense2], axis=1, name='concat')
            dense1 = tf.layers.dense(dense1, units=64, activation=tf.nn.relu, kernel_initializer=w_initializer, name='dense1')
            dense1 = tf.layers.batch_normalization(dense1, name='bn2')
            dense2 = tf.layers.dense(dense1, units=64, activation=tf.nn.relu, kernel_initializer=w_initializer, name='dense2')
            self.critic_target = tf.layers.dense(dense2, 1, kernel_initializer=w_initializer, name='critic_eval')

        # ------------------ build actor eval net ----------------------- #
        with tf.variable_scope('actor_eval_net'):
            dense1 = tf.layers.dense(self.p_s, 256, tf.nn.relu, kernel_initializer=w_initializer, name='dense1')
            dense2 = tf.layers.dense(dense1, 256, tf.nn.relu, kernel_initializer=w_initializer, name='dense2')
            dense3 = tf.layers.dense(dense2, 1, tf.nn.tanh, kernel_initializer=w_initializer, name='dense3')
            self.actor_eval = tf.multiply(dense3, self.upper_bound, name='actor_eval_output')

        # ------------------ build actor target net --------------------- #
        with tf.variable_scope('actor_target_net'):
            dense1 = tf.layers.dense(self.p_s_, 256, tf.nn.relu, kernel_initializer=w_initializer, name='dense1')
            dense2 = tf.layers.dense(dense1, 256, tf.nn.relu, kernel_initializer=w_initializer, name='dense2')
            dense3 = tf.layers.dense(dense2, 1, tf.nn.tanh, kernel_initializer=w_initializer, name='dense3')
            self.actor_target = tf.multiply(dense3, self.upper_bound, name='actor_target_output')

        self.trainable_actor_parameters = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'actor_eval_net')
        with tf.variable_scope('critic_loss'):
            self.critic_loss = tf.reduce_mean(tf.squared_difference(self.critic_eval, self.p_target_critic, name='TD_error'))

        with tf.variable_scope("train_critic"):
            self.train_critic_op = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(self.critic_loss)

        with tf.variable_scope("train_actor"):
            parameters_gradient = tf.gradients(self.actor_eval, self.trainable_actor_parameters, -self.p_q_gradient, name='actor_gradients')
            self.train_actor_op = tf.train.AdamOptimizer().apply_gradients(zip(parameters_gradient, self.trainable_actor_parameters))

    def learn(self):
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size, replace=False)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]
        n_features_tensor = [self.batch_size] + [dim for dim in self.feature_shape]
        # ------------------------------------------------------------------- #
        s = batch_memory[:, :self.n_features]
        a = np.reshape(batch_memory[:, self.n_features], (self.batch_size, 1))
        s_ = batch_memory[:, -self.n_features:]
        r = np.reshape(batch_memory[:, self.n_features + 1], (self.batch_size, 1))
        d = batch_memory[:, self.n_features + 2]

        # -------------------Training Critic Network (Q network)------------- #
        target_actions = self.sess.run(self.actor_target, feed_dict={self.p_s_: s_})
        critic_target_output = self.sess.run(self.critic_target, feed_dict={self.p_s_: s_, self.p_a_: target_actions})

        y = r + self.gamma * critic_target_output
        critic_loss, _ = self.sess.run([self.critic_loss, self.train_critic_op], feed_dict={self.p_s: s, self.p_a: a, self.p_target_critic: y})

        # ------Training Actor Network (double policy gradient network)------ #
        actions = self.sess.run(self.actor_eval, feed_dict={self.p_s: s})
        critic_value = self.sess.run(self.critic_eval, feed_dict={self.p_s: s, self.p_a: actions})

        # 要最大化值，在前面加一个负号来最小化即可
        _ = self.sess.run(self.train_actor_op, feed_dict={self.p_s: s, self.p_q_gradient: critic_value})

    def _store_transition(self, s, a, r, s_, d):
        transition = np.hstack((np.reshape(s, [-1]), [a, r, int(d)], np.reshape(s_, [-1])))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, s):
        observation = np.expand_dims(s, axis=0)
        action = np.squeeze(self.sess.run(self.actor_eval, feed_dict={self.p_s: observation}))
        sampled_action = action
        legal_action = np.clip(sampled_action, self.lower_bound, self.upper_bound)
        return legal_action

    def train(self):
        step = 0
        for i in range(1000):
            observation = self.env.reset()
            sum_reward = 0.
            e_step = 0
            while True:
                self.env.render()
                action = np.array([self.choose_action(observation)])
                s_, r, done, info = self.env.step(action)
                sum_reward += r
                self._store_transition(observation, action, r, s_, done)
                if self.memory_counter >= self.memory_size:
                    self.learn()
                if done:
                    break
                step += 1
                e_step += 1
                if step % 50 == 0:
                    self.update_target()
            print("Episode: %d, AVG_reward: %.5f" % (i, sum_reward / e_step))


if __name__ == '__main__':
    agent = Agent(env)
    print(env.action_space)
    agent.train()
