import numpy as np
# import tensorflow.compat.v1 as tf
import tensorflow._api.v2.compat.v1 as tf
import gym
import matplotlib.pyplot as plt

tf.disable_eager_execution()
tf.disable_v2_behavior()


class DoubleDQN:
    def __init__(self,
                 n_actions, n_features, learning_rate=0.005, reward_decay=0.9, replace_decay=0.9, e_greedy=0.9,
                 replace_target_iter=200, memory_size=3000, batch_size=32, e_greedy_increment=None,
                 output_graph=False, double_q=True, sess: tf.Session = None):
        self.n_actions = n_actions
        self.n_features = n_features
        self.learning_rate = learning_rate
        self.gamma = reward_decay
        self.replace_decay = replace_decay
        self.replace_target_iter = replace_target_iter
        self.epsilon_max = e_greedy
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.e_greedy_increment = e_greedy_increment
        self.output_graph = output_graph
        self.double_q = double_q
        self.memory_counter = 0
        self.learn_step_counter = 0
        self.memory = np.zeros((self.memory_size, self.n_features * 2 + 2))
        self.epsilon = 0 if self.e_greedy_increment is not None else self.epsilon_max
        self._build_net()

        e_params = tf.get_collection('eval_net_params')
        t_params = tf.get_collection('target_net_params')
        with tf.variable_scope("assign_op"):
            self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        if sess is None:
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())
        else:
            self.sess = sess

        if output_graph:
            tf.summary.FileWriter("./logs/", self.sess.graph)
        self.cost_his = []  # 损失函数历史记录

    def _build_net(self):
        # Building the structure of neural network.
        def build_layer(s, c_names, n_l1, n_l2, w_initializer, b_initializer):
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(s, w1) + b1)
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, n_l2], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, n_l2], initializer=b_initializer, collections=c_names)
                l2 = tf.nn.relu(tf.matmul(l1, w2) + b2)
            with tf.variable_scope('l3'):
                w3 = tf.get_variable('w3', [n_l2, self.n_actions], initializer=w_initializer, collections=c_names)
                b3 = tf.get_variable('b3', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                l3 = tf.nn.relu(tf.matmul(l2, w3) + b3)
            return l3

        # Building the evaluate net

        self.state = tf.placeholder(tf.float32, [None, self.n_features], name='state')
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='q_target')  # expect output

        with tf.variable_scope('eval_net'):
            c_names, n_l1, n_l2, w_initializer, b_initializer = ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 64, 64, tf.random_normal_initializer(
                0.0, 0.3), tf.random_normal_initializer(0., 0.3)
            self.q_eval = build_layer(self.state, c_names, n_l1, n_l2, w_initializer, b_initializer)

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))

        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)
        # Building the target net.
        self.state_ = tf.placeholder(tf.float32, [None, self.n_features], name='state_')
        with tf.variable_scope('target_net'):
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
            self.q_next = build_layer(self.state_, c_names, n_l1, n_l2, w_initializer, b_initializer)

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, current_state):
        current_state = current_state[np.newaxis, :]
        action_value = self.sess.run(self.q_eval, feed_dict={self.state: current_state})
        action = np.argmax(action_value)
        if not hasattr(self, 'q'):
            self.q = []
            self.running_q = 0.
        self.running_q = self.running_q * 0.99 + 0.01 * np.max(action_value)
        self.q.append(self.running_q)
        if np.random.uniform() > self.epsilon:
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            print("target params replaced!")

        if self.memory_counter > self.memory_size:
            # 当存入replay_buffer中的数据量超过memory_size时,从memory_size中选取,防止越界
            sample_index = np.random.choice(self.memory_size, self.batch_size)
        else:
            # 当存入replay_buffer中的数据量小于memory_size是,从memory_counter中取,防止取到无效数据
            sample_index = np.random.choice(self.memory_counter, self.batch_size)
        batch_memory = self.memory[sample_index, :]
        q_eval = self.sess.run(self.q_eval, feed_dict={self.state: batch_memory[:, :self.n_features]})  # 传入当前状态
        q_next = self.sess.run(self.q_next, feed_dict={self.state_: batch_memory[:, -self.n_features:]})  # 传入下一个状态

        batch_index = np.arange(self.batch_size, dtype=np.int32)

        q_target = q_eval.copy()  # 输出每个action对应的q value
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]

        if self.double_q:
            # double q learning 中选择最大的q是从q_eval中选择, 而不是从q_target中选择.
            max_act = np.argmax(q_eval, axis=1)
            selected_q_next = q_next[batch_index, max_act]
        else:
            max_act = np.argmax(q_next, axis=1)
            selected_q_next = q_next[batch_index, max_act]
        q_target[batch_index, eval_act_index] = reward + self.gamma * selected_q_next

        _, loss_value = self.sess.run([self._train_op, self.loss], feed_dict={self.q_target: q_target,
                                                                              self.state: batch_memory[:,
                                                                                          :self.n_features]})
        self.epsilon += self.e_greedy_increment if self.epsilon < self.epsilon_max else 0
        self.cost_his.append(loss_value)
        self.learn_step_counter += 1


class RunClass:
    def __init__(self):
        self.sess = tf.Session()
        self.action_space = 11
        self.memory_size = 3000
        self.n_features = 3
        self.env = gym.make('Pendulum-v0')
        self.env = self.env.unwrapped
        self.env.seed(1)
        self.build_learning_model()

    def build_learning_model(self):
        # with tf.variable_scope("Natural-DQN"):
        #     self.natural_dqn = DoubleDQN(self.action_space, self.n_features, memory_size=self.memory_size,
        #                                  e_greedy_increment=0.001, double_q=False, sess=self.sess, output_graph=True)

        with tf.variable_scope("Double-DQN"):
            self.double_dqn = DoubleDQN(self.action_space, self.n_features, memory_size=self.memory_size,
                                        double_q=True, sess=self.sess, e_greedy_increment=0.001, output_graph=True)
        self.sess.run(tf.global_variables_initializer())

    def train(self, RL: DoubleDQN):
        total_step = 0
        observation = self.env.reset()
        while True:
            action = RL.choose_action(observation)
            #             将action转换到[-2, 2]之间
            f_action = (action - (self.action_space - 1) / 2) / ((self.action_space - 1) / 4)
            observation_, reward, done, info = self.env.step(np.array([f_action]))
            reward /= 10
            RL.store_transition(observation, action, reward, observation_)
            if total_step > self.memory_size:
                RL.learn()
            if total_step - self.memory_size > 20000:
                break
            observation = observation_
            total_step += 1
        return RL.q

    def run(self):
        # q_natural = self.train(self.natural_dqn)
        q_double = self.train(self.double_dqn)
        # plt.plot(np.array(q_natural), c='r', label='natural')
        plt.plot(np.array(q_double), c='b', label='double')
        plt.legend(loc='best')
        plt.ylabel('Q eval')
        plt.xlabel('training steps')
        plt.grid()
        plt.show()


if __name__ == '__main__':
    run_class = RunClass()
    run_class.run()
