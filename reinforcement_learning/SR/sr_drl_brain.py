import numpy as np
import utils
from gridworld import SimpleGrid
import tensorflow as tf
import os
import shutil

grid_size = 7
pattern = "four_rooms"
env = SimpleGrid(grid_size, block_pattern=pattern, obs_mode="index")
env.reset(agent_pos=[0, 0], goal_pos=[0, grid_size - 1])
# plt.imshow(env.grid)
# plt.show()
sess = tf.Session()


# sess.run(tf.global_variables_initializer())

def _build_layer(input_dim, output_dim, input_data, c_name, layer_name, w_name='w', bias_name='b', has_activate=True):
    with tf.variable_scope(layer_name):
        w = tf.get_variable(w_name, [input_dim, output_dim], dtype=tf.float32, initializer=tf.truncated_normal_initializer(),
                            collections=c_name)
        b = tf.get_variable(bias_name, [output_dim], dtype=tf.float32, initializer=tf.truncated_normal_initializer(), collections=c_name)
        if has_activate:
            l = tf.nn.relu(tf.add(tf.matmul(input_data, w), b))
        else:
            l = tf.add(tf.matmul(input_data, w), b)
    return l


class TabularSuccessorAgent(object):
    def __init__(self, n_state, n_action, learning_rate, gamma, is_double_q=True):
        self.n_state = n_state
        self.n_action = n_action
        self.is_double_q = is_double_q
        self.fai_s_size = 512
        # shape: (state_size, action_size)

        self.w = np.zeros([n_state])
        self.learning_rate = learning_rate
        self.gamma = gamma

        self.replay_buffer = []
        self.state = tf.placeholder(tf.float32, [None, self.n_state])
        self.state_ = tf.placeholder(tf.float32, [None, self.n_state])

        self.rs_p = tf.placeholder(tf.float32, [None, 1])
        self.sess = tf.Session()

        self.eval_collection_name = ['eval_net_collection', tf.GraphKeys.GLOBAL_VARIABLES]
        self.target_collection_name = ['target_net_collection', tf.GraphKeys.GLOBAL_VARIABLES]

        shutil.rmtree("./log")
        os.mkdir("./log")
        with tf.variable_scope('eval_net'):
            eval_s_hat, eval_r_s, eval_M = self._build_net(self.eval_collection_name)
        with tf.variable_scope('target_net'):
            target_s_hat, target_r_s, target_M = self._build_net(self.target_collection_name)
        with tf.variable_scope('assign_op'):
            e_params = tf.get_collection('eval_collection_name')
            t_params = tf.get_collection('target_net_collection')
            self.assign_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        tf.summary.FileWriter("./log", self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

    def _build_net(self, c_name):
        # 这里我们将一维的state进行one-hot编码处理, 因此输入的维度是1*n_state
        with tf.variable_scope('f_theta'):
            l1 = _build_layer(self.n_state, 128, self.state, c_name, 'l1')
            l2 = _build_layer(128, 512, l1, c_name, 'l2')
            l3 = _build_layer(512, 1024, l2, c_name, 'l3')
            l4 = _build_layer(1024, 512, l3, c_name, 'fai_s')
            self.fai_s = l4
        with tf.variable_scope('g_theta_'):
            dl1 = _build_layer(512, 1024, l4, c_name, 'dl1')
            dl2 = _build_layer(1024, 512, dl1, c_name, 'dl2')
            dl3 = _build_layer(512, 128, dl2, c_name, 'dl3')
            s_hat = _build_layer(128, self.n_state, dl3, c_name, 'dl4')
        with tf.variable_scope('R_s'):
            r_s = _build_layer(512, 1, l4, c_name, 'r', has_activate=False)
        with tf.variable_scope("mu_alpha"):
            M = np.stack([np.zeros(self.fai_s_size, object) for i in range(self.n_action)])
            for i in range(self.n_action):
                miu_layer = _build_layer(256, 512, _build_layer(512, 256, _build_layer(512, 512, l4, c_name, 'm%s-1' % i, 'miu_alpha_l1_w', 'miu_alpha_l1_b'),
                                                                c_name, 'm%s-2' % i, 'miu_alpha_l2_w', 'miu_alpha_l2_b'), c_name, 'm%s-3' % i, 'miu_alpha_l3_w',
                                         'miu_alpha_l3_b')
                M[i] = miu_layer

        return s_hat, r_s, M

        loss1 = tf.reduce_mean(tf.squared_difference(self.rs_p, r_s))
        loss2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.state, logits=s_hat))

        a, M1 = self.choose_action(self.state, 0)
        a_, M2 = self.choose_action(self.state_, 0)
        loss3 = tf.squared_difference(l4 + self.gamma*M2[a_] - M1[a])


    def choose_action(self, state, epsilon=1.):
        # 计算q值
        M = np.zeros([self.n_action, self.fai_s_size])
        for i in range(self.n_action):
            M[i] = sess.run(M[i], feed_dict={self.state, state})
        w = sess.graph.get_tensor_by_name('eval_net/R_s/r/w:0')
        q_ = tf.matmul(w, tf.Variable(M))
        if np.random.uniform(0, 1) < epsilon:
            action = np.random.randint(self.n_action)
        else:
            action = np.argmax(q_)
        return action, M

    def update_rs(self, current_exp):

        # A simple update rule. current_exp's shape: [state, a, next_state, r, done]
        s_ = current_exp[2]
        r = current_exp[3]
        error = r - self.w[s_]
        self.w[s_] += self.learning_rate * error  # w相当于公式中对R(s)的更新
        return error

    def update_sr(self, current_exp, next_exp):
        # SARSA TD learning rule
        # update the M(s, s', a)
        s = current_exp[0]  # current state
        s_a = current_exp[1]  # choosed action
        s_ = current_exp[2]  # next state
        s_a_1 = next_exp[1]  # next state choosed action
        r = current_exp[3]  # reward in current state
        d = current_exp[4]  # wheather the current state is terminal
        I = utils.onehot(s, env.state_size)  # transform current state to one-hot vector
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

experiences = []  # replay buffer [(current_state), action, (next_state), reward, done]  == experience.shape
test_experiences = []
test_lengths = []
lifetime_td_errors = []

for i in range(episodes):
    # Train phase
    agent_start = [0, 0]
    if i < episodes // 2:
        goal_pos = [0, grid_size - 1]
    else:
        if i == episodes // 2:
            print("\nSwitched reward locations")
        goal_pos = [grid_size - 1, grid_size - 1]
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

# fig = plt.figure(figsize=(10, 6))
#
# ax = fig.add_subplot(2, 2, 1)
# ax.plot(lifetime_td_errors)
# ax.set_title("TD Error")
# ax = fig.add_subplot(2, 2, 2)
# ax.plot(test_lengths)
# ax.set_title("Episode Lengths")
# plt.show()
