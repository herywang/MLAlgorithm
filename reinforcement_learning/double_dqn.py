import numpy as np
import tensorflow.compat.v1 as tf


class DoubleDQN:
    def __init__(self,
                 n_actions, n_features, learning_rate=0.005, reward_decay=0.9, replace_decay=0.9, e_greedy=0.9,
                 replace_target_iter=200, memory_size=3000, batch_size=32, e_greedy_increment=None,
                 output_graph=False, double_q=True, sess=None):
        self.n_actions = n_actions
        self.n_features = n_features
        self.learning_rate = learning_rate
        self.reward_decay = reward_decay
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

        e_params = tf.get_collection('eval_net_params')
        t_params = tf.get_collection('target_net_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        if sess is None:
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())
        else:
            self.sess = sess

        if output_graph:
            tf.summary.FileWriter("./logs/", self.sess.graph)
        self.cost_his = []

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

        #       Building the evaluate net

        self.state = tf.placeholder(tf.float32, [None, self.n_features], name='state')
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='q_target')  # expect output

        with tf.variable_scope('eval_net'):
            c_names, n_l1, n_l2, w_initializer, b_initializer = ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 64, 64, tf.random_normal_initializer(
                0.0, 0.3), tf.random_normal_initializer(0., 0.3)
            self.q_eval_net_output = build_layer(self.state, n_l1, n_l2, w_initializer, b_initializer)

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval_net_output))
        with tf.variable_scope('train'):
            self.train = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)

        #       Building the target net.

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
        action_value = self.sess.run(self.q_eval_net_output, feed_dict={self.state: current_state})
        action = np.argmax(action_value)

        if not hasattr(self, 'q'):
            self.q = []
            self.running_q = 0.
        self.running_q = self.running_q * 0.99 + 0.01 * np.max(action_value)
        self.q.append(self.running_q)

        if np.random.uniform() > self.epsilon:
            action = np.random.randint(0, self.n_actions)
        return action

