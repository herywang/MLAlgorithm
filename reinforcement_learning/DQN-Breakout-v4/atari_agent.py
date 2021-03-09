import tensorflow as tf
import gym
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle

tf.compat.v1.disable_eager_execution()
tf.compat.v1.disable_v2_behavior()

env = gym.make('Breakout-v4')


class AgentDQN():
    def __init__(self, env, args):
        tf.reset_default_graph()
        self.env = env
        self.get_path()  # 创建模型以及训练历史数据保存的路径
        self.double_q = True
        self.n_actions = self.env.action_space.n
        self.feature_shape = self.env.observation_space.shape
        self.n_features = np.prod(np.array(self.feature_shape))  # 内积, 获取到特征数量
        self.memory_size = 10000
        self.gamma = 0.99
        self.batch_size = 32
        self.n_steps = 7e6
        self.epsilon = 1.0
        self.epsilon_min = 0.07
        self.epsilon_decrement = (self.epsilon - self.epsilon_min) / 100000.
        # 每4步探索学习一次， 防止帧率过大，导致很多接近重复的frame
        self.learn_every_n_step = 4
        self.save_every_n_episode = 100
        # 前10000次采用随机探索，不学习
        self.start_learning_after_n_step = 10000
        # 每1000次更新一次target_net的参数
        self.update_network_after_n_step = 1000
        self.reward_his = []
        self.memory = np.zeros((self.memory_size, 3 + self.n_features * 2))
        # -----------------------------model-----------------------------#
        self.build_model()
        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')
        with tf.variable_scope('soft_replacement'):
            self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep=10)

        load = bool(0)
        if args.test_dqn or load:
            print("Loading trained model: " + self.model_path)
            if load: self.reward_his = pickle.load(open(self.reward_his_path, 'rb'))
            try:
                self.saver.restore(self.sess, save_path=tf.train.latest_checkpoint(self.model_path))
            except:
                self.saver.restore(self.sess, save_path=os.path.join(self.model_path, 'model_dqn-25581'))

    def build_model(self):
        n_features_tensor = [None] + [dim for dim in self.feature_shape]
        self.s = tf.placeholder(tf.float32, n_features_tensor, name='s')
        self.s_ = tf.placeholder(tf.float32, n_features_tensor, name='s_')
        self.a = tf.placeholder(tf.float32, [None, ], name='a')
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')

        # ------------------ initializers ------------------ #
        from tensorflow.python.ops.init_ops import VarianceScaling

        def lecun_normal(seed=None):
            return VarianceScaling(scale=1., mode='fan_in', distribution='truncated_normal', seed=seed)

        def lecun_uniform(seed=None):
            return VarianceScaling(scale=1., mode='fan_in', distribution='uniform', seed=seed)

        w_initializer = lecun_normal()
        b_initializer = tf.zeros_initializer()

        # ------------------ build evaluate_net ------------------ #
        with tf.variable_scope('eval_net'):
            e_conv1 = tf.layers.conv2d(self.s, filters=32,
                                       kernel_size=[5, 5],
                                       strides=[4, 4],
                                       padding='same',
                                       activation=tf.nn.relu6,
                                       kernel_initializer=w_initializer,
                                       name='e_conv1')
            e_conv2 = tf.layers.conv2d(e_conv1, filters=64,
                                       kernel_size=[5, 5],
                                       strides=[2, 2],
                                       padding='same',
                                       activation=tf.nn.relu6,
                                       kernel_initializer=w_initializer,
                                       name='e_conv2')
            e_conv3 = tf.layers.conv2d(e_conv2, filters=64,
                                       kernel_size=[3, 3],
                                       strides=[1, 1],
                                       padding='same',
                                       activation=tf.nn.relu6,
                                       kernel_initializer=w_initializer,
                                       name='e_conv3')
            e_flat = tf.contrib.layers.flatten(e_conv3)
            e_dense1 = tf.layers.dense(inputs=e_flat,
                                       units=512,
                                       activation=tf.nn.relu,
                                       kernel_initializer=w_initializer,
                                       bias_initializer=b_initializer,
                                       name='e_dense1')
            self.q_eval = tf.layers.dense(inputs=e_dense1,
                                          units=self.n_actions,
                                          activation=None,
                                          kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer,
                                          name='q_eval')  # q_new shape: (batch_size, n_actions)
        # ------------------ build target net ------------------ #
        with tf.variable_scope('target_net'):
            t_conv1 = tf.layers.conv2d(inputs=self.s_,
                                       filters=32,
                                       kernel_size=[5, 5],
                                       strides=(4, 4),
                                       padding="same",
                                       activation=tf.nn.relu6,
                                       kernel_initializer=w_initializer,
                                       name='t_conv1')
            t_conv2 = tf.layers.conv2d(inputs=t_conv1,
                                       filters=64,
                                       kernel_size=[5, 5],
                                       strides=(2, 2),
                                       padding="same",
                                       activation=tf.nn.relu6,
                                       kernel_initializer=w_initializer,
                                       name='t_conv2')
            t_conv3 = tf.layers.conv2d(inputs=t_conv2,
                                       filters=64,
                                       kernel_size=[3, 3],
                                       strides=(1, 1),
                                       padding="same",
                                       activation=tf.nn.relu6,
                                       kernel_initializer=w_initializer,
                                       name='t_conv3')
            t_flat = tf.contrib.layers.flatten(t_conv3)
            t_dense1 = tf.layers.dense(inputs=t_flat,
                                       units=512,
                                       activation=tf.nn.relu,
                                       kernel_initializer=w_initializer,
                                       bias_initializer=b_initializer,
                                       name='t_dense1')
            self.q_old = tf.layers.dense(inputs=t_dense1,
                                         units=self.n_actions,
                                         activation=None,
                                         kernel_initializer=w_initializer,
                                         bias_initializer=b_initializer,
                                         name='q_target')

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval, name='TD_error'))

        with tf.variable_scope("train"):
            self.train_op = tf.train.RMSPropOptimizer(learning_rate=0.0001, decay=0.99).minimize(self.loss)

    def store_transition(self, s, a, r, d, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        transition = np.hstack((np.reshape(s, [-1]), [a, r, int(d)], np.reshape(s_, [-1])))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size, replace=False)  # sample batch memory from all memory
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]
        n_features_tensor = [self.batch_size] + [dim for dim in self.feature_shape]
        # -----------------------------------------------------------------------------------#
        s = np.reshape(batch_memory[:, :self.n_features], n_features_tensor)
        actions = batch_memory[:, self.n_features].astype(int)
        rewards = batch_memory[:, self.n_features + 1]
        done = batch_memory[:, self.n_features + 2]
        s_ = np.reshape(batch_memory[:, -self.n_features:], n_features_tensor)
        # -----------------------------------------------------------------------------------#

        q_eval, q_old = self.sess.run([self.q_eval, self.q_old], feed_dict={self.s: s, self.s_: s_})
        q_target_ = q_old.copy()
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        if self.double_q:
            q_new4next = self.sess.run(self.q_eval, feed_dict={self.s: s_})
            maxactionnext = np.argmax(q_new4next, axis=1)
            selected_q_target = q_old[batch_index, maxactionnext]
        else:
            selected_q_target = np.max(q_old, axis=1)
        q_target_[batch_index, actions] = rewards + (1 - done) * self.gamma * selected_q_target  # change q_target w.r.t q_new's action

        _, loss = self.sess.run([self.train_op, self.loss], feed_dict={self.s: s, self.q_target: q_target_})
        return loss

    def train(self):
        global action
        episode = 0
        step = 0
        loss = 9.999
        rwd_avg_max = 0

        while step < self.n_steps:

            observation = self.env.reset()
            done = False
            episode_reward = 0.0

            while not done:
                action = self.make_action(observation, test=True)
                observation_, reward, done, info = self.env.step(action)
                episode_reward += reward

                self.store_transition(observation, action, reward, done, observation_)
                if (step > self.start_learning_after_n_step) and (step % self.learn_every_n_step == 0):
                    loss = self.learn()
                if (step > self.start_learning_after_n_step) and (step % self.update_network_after_n_step == 0):
                    self.sess.run(self.target_replace_op)

                print('Step: %i, Episode: %i, Action:%i, Reward:%.2f, Epsilon: %.5f, Loss:%.5f' % (step, episode, action, reward, self.epsilon, loss), end='\r')
                self.epsilon = self.epsilon - self.epsilon_decrement if self.epsilon > self.epsilon_min else self.epsilon_min  # decreasing epsilon
                observation = observation_
                step += 1

            episode += 1
            self.reward_his.append(episode_reward)
            if step < 1000000:
                # 前1000,000步每100步保存一次模型数据
                if episode % self.save_every_n_episode == 0:
                    pickle.dump(self.reward_his, open(self.reward_his_path, 'wb'), True)
                    self.saver.save(self.sess, os.path.join(self.model_path, 'model-dqn'), global_step=episode)
            else:
                rwd_avg = np.mean(self.reward_his[-20:])
                if rwd_avg > rwd_avg_max:
                    pickle.dump(self.reward_his, open(self.reward_his_path, 'wb'), True)
                    self.saver.save(self.sess, os.path.join(self.model_path, 'model-dqn'), global_step=episode)
                    rwd_avg_max = rwd_avg
                    print("Saving best model with avg reward: ", rwd_avg_max)
                if episode % self.save_every_n_episode == 0:
                    pickle.dump(self.reward_his, open(self.reward_his_path, 'wb'), True)
            print('Step: %i/%i,  Episode: %i,  Action: %i,  Episode Reward: %.0f,  Epsilon: %.2f, Loss: %.5f' % (
                step, self.n_steps, episode, action, episode_reward, self.epsilon, loss))

    def make_action(self, observation, test):
        observation = np.expand_dims(observation, axis=0)
        if test: self.epsilon = 0.01  # 如果是测试环境, 探索概率设置为1%
        if np.random.uniform() > self.epsilon:
            action_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(action_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action  # self.env.get_random_action()

    def get_path(self):
        directory = "./model"
        try:
            if not os.path.exists(directory):
                os.mkdir(directory)
        except:
            print("Filed to create result directory.")
            directory = "./"
        self.reward_his_path = os.path.join(directory, "reward_his_dqn.pkl")
        self.model_path = directory
