import tensorflow as tf
import numpy as np
import os
import pickle

tf.disable_v2_behavior()
tf.disable_eager_execution()


class AgentDSR(object):
    def __init__(self, env, n_steps=3000, output_graph=False):
        self.env = env  # environment
        self.n_steps = n_steps  # Training steps.
        self.output_graph = output_graph
        self.n_actions = self.env.action_space.n
        self.feature_shape = self.env.observation_space.shape
        self.n_features = np.prod(np.array(self.feature_shape))
        self.save_every_n_episode = 100
        self.start_learning_after_n_step = 3000
        self.memory_size = 256
        self.memory = np.zeros((self.memory_size, 3 + self.n_features * 2))
        self.memory_counter = 0
        self.batch_size = 32
        self.gamma = 0.9
        self.epsilon = 1.0
        self.epsilon_min = 0.07
        self.epsilon_decrement = (self.epsilon - self.epsilon_min) / 8000.
        self.reward_his = []

        self.fi_size = 64
        self.sess = tf.Session()
        self.__get_model_save_path()
        self.__build_model()

        self.loss1_p = tf.placeholder(tf.float32, name='autoencoder_loss')
        self.loss2_p = tf.placeholder(tf.float32, name='reward_loss')
        self.loss3_p = tf.placeholder(tf.float32, name='mu_loss')
        self.loss_p = tf.placeholder(tf.float32, name='all_loss')
        if self.output_graph:
            self.summary = [tf.summary.scalar('loss1_autoencoder_loss', self.loss1_p),
                            tf.summary.scalar('loss2_reward_loss', self.loss2_p),
                            tf.summary.scalar('loss3_mu_loss', self.loss3_p),
                            tf.summary.scalar('loss', self.loss_p)]
            self.merged = tf.summary.merge_all()
            self.__ouput_graph()

        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep=10)

    def __build_model(self):
        from tensorflow.python.ops.init_ops import VarianceScaling

        def lecun_normal(seed=None):
            return VarianceScaling(seed=seed)

        w_initializer = lecun_normal()

        self.s_p = tf.placeholder(tf.float32, [None, self.n_features], 'state')
        self.decoded_s_p = tf.placeholder(tf.float32, [None, self.n_features], "decoded_state")
        self.r_p = tf.placeholder(tf.float32, [None, 1], 'r')
        self.fi_eval_p = tf.placeholder(tf.float32, [None, self.fi_size], name='fi')
        self.mu_next_p = tf.placeholder(tf.float32, [None, self.fi_size], name='mu_next')
        self.action_index_p = tf.placeholder(tf.int32, name='action_index')

        self.mus_eval = []
        self.mus_target = []

        with tf.variable_scope('encoder'):
            en_1 = tf.layers.dense(self.s_p, 32, tf.nn.relu6, kernel_initializer=w_initializer, name='en_1')
            en_2 = tf.layers.dense(en_1, 64, tf.nn.relu6, kernel_initializer=w_initializer, name='en_2')
            en_3 = tf.layers.dense(en_2, 64, tf.nn.relu6, kernel_initializer=w_initializer, name='en_3')
            self.encoded = tf.layers.dense(en_3, self.fi_size, tf.nn.softmax, kernel_initializer=w_initializer, name='encoded')

        with tf.variable_scope('decoder'):
            de_1 = tf.layers.dense(self.encoded, 64, tf.nn.relu6, kernel_initializer=w_initializer, name='de_1')
            de_2 = tf.layers.dense(de_1, 64, tf.nn.relu6, kernel_initializer=w_initializer, name='de_2')
            de_3 = tf.layers.dense(de_2, 32, tf.nn.relu6, kernel_initializer=w_initializer, name='de_3')
            self.decoded = tf.layers.dense(de_3, self.n_features, kernel_initializer=w_initializer, name='decoded')

        with tf.variable_scope('reward'):
            self.R = tf.layers.dense(self.fi_eval_p, 1, tf.nn.relu6, kernel_initializer=w_initializer, name='R')

        # for i in range(self.n_actions):
        #     with tf.variable_scope('mu_%d' % i):
        #         mu_1 = tf.layers.dense(self.fi_eval_p, 64, tf.nn.relu6, kernel_initializer=w_initializer, name='action_%d_mu1' % i)
        #         mu_2 = tf.layers.dense(mu_1, 64, tf.nn.relu6, kernel_initializer=w_initializer, name='action_%d_mu2' % i)
        #         mu_3 = tf.layers.dense(mu_2, self.fi_size, tf.nn.relu6, kernel_initializer=w_initializer, name='action_%d_mu3' % i)
        #         self.mus_eval.append(mu_3)


        with tf.variable_scope('encoder_loss'):
            self.loss1 = tf.reduce_mean(tf.squared_difference(self.decoded, self.decoded_s_p, name='loss1'))

        with tf.variable_scope('reward_loss'):
            self.loss2 = tf.reduce_mean(tf.squared_difference(self.R, self.r_p, name='loss2'))

        with tf.variable_scope('mu_loss'):
            self.loss3 = tf.reduce_mean(tf.squared_difference(self.fi_eval_p + self.gamma * self.mu_next_p, tf.gather(self.mus_eval, self.action_index_p), name='loss3'))

        self.loss = self.loss1 + self.loss2 + self.loss3

        with tf.variable_scope('train'):
            self.train_op1 = tf.train.RMSPropOptimizer(0.0001, 0.9).minimize(self.loss1)
            self.train_op2 = tf.train.RMSPropOptimizer(0.0001, 0.9).minimize(self.loss2)
            self.train_op3 = tf.train.RMSPropOptimizer(0.0001, 0.9).minimize(self.loss3)

    def learn(self, step):
        """
        :return: Learning network's parameters.
        """
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size, replace=False)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]
        n_features_tensor = [self.batch_size] + [dim for dim in self.feature_shape]
        # ------------------------------------------------------------------- #
        s = np.reshape(batch_memory[:, :self.n_features], n_features_tensor)
        actions = batch_memory[:, self.n_features].astype(int)
        rewards = np.reshape(batch_memory[:, self.n_features + 1], (self.batch_size, 1))
        s_ = batch_memory[:, -self.n_features:]
        w = self.sess.run(self.get_w())
        # ------------------------------------------------------------------- #
        # autoencoder loss.
        loss1, _ = self.sess.run([self.loss1, self.train_op1], feed_dict={self.s_p: s, self.decoded_s_p: s})

        fi = self.sess.run(self.encoded, feed_dict={self.s_p: s})
        fi_ = self.sess.run(self.encoded, feed_dict={self.s_p: s_})

        # reward loss.
        loss2, _ = self.sess.run([self.loss2, self.train_op2], feed_dict={self.fi_eval_p: fi, self.r_p: rewards})

        # mu loss.
        sum_loss3 = 0.
        for i in range(actions.shape[0]):
            # batch iteration.
            mus_ = self.sess.run(self.mus_eval, feed_dict={self.fi_eval_p: np.expand_dims(fi_[i], axis=0)})
            max_action_index_s_ = np.argmax(np.matmul(mus_, w))
            loss3, _ = self.sess.run([self.loss3, self.train_op3], feed_dict={self.fi_eval_p: np.expand_dims(fi[i],axis=0),
                                                                              self.mu_next_p: mus_[max_action_index_s_],
                                                                              self.action_index_p: actions[i]})
            sum_loss3 += loss3
        mean_loss3 = sum_loss3 / actions.shape[0]
        if self.output_graph:
            [s_loss1, s_loss2, s_loss3, s_loss] = self.sess.run(self.summary, feed_dict={self.loss1_p: loss1, self.loss2_p: loss2,
                                                                                         self.loss3_p: mean_loss3, self.loss_p: loss1 + loss2 + mean_loss3})
            self.train_writer.add_summary(s_loss1, step)
            self.train_writer.add_summary(s_loss2, step)
            self.train_writer.add_summary(s_loss3, step)
            self.train_writer.add_summary(s_loss, step)
            self.train_writer.flush()

        return loss1, loss2, mean_loss3

    def train(self):
        global action
        episode = 0  # 训练多少轮
        step = 0
        reward_avg_max = 0
        loss3 = 9.999

        while step < self.n_steps:
            observation = self.env.reset()
            done = False
            episode_reward = 0.0
            while not done:
                self.env.render()
                action = self.choice_action(observation, test=False)
                observation_, reward, done, info = self.env.step(action)
                episode_reward += reward
                self.store_trainsition(observation, action, reward, done, observation_)
                if step > self.start_learning_after_n_step:
                    loss1, loss2, loss3 = self.learn(step)
                    print('Step: %i, Episode: %i, Action:%i, Reward:%.2f, Epsilon: %.5f, Loss1:%.5f, Loss2:%.5f, Loss3:%.5f'
                          % (step, episode, action, reward, self.epsilon, loss1, loss2, loss3), end='\r')
                    self.epsilon = self.epsilon - self.epsilon_decrement if self.epsilon > self.epsilon_min else self.epsilon_min  # decreasing epsilon
                observation = observation_
                step += 1
            episode += 1
            self.reward_his.append(episode_reward)
            if step < 2500:
                # 前1000,000步每100步保存一次模型数据
                if episode % self.save_every_n_episode == 0:
                    pickle.dump(self.reward_his, open(self.reward_his_path, 'wb'), True)
                    self.saver.save(self.sess, os.path.join(self.model_path, 'model-dqn'), global_step=episode)
            else:
                rwd_avg = np.mean(self.reward_his[-20:])
                if rwd_avg > reward_avg_max:
                    pickle.dump(self.reward_his, open(self.reward_his_path, 'wb'), True)
                    self.saver.save(self.sess, os.path.join(self.model_path, 'model-dqn'), global_step=episode)
                    reward_avg_max = rwd_avg
                    print("Saving best model with avg reward: ", reward_avg_max)
                if episode % self.save_every_n_episode == 0:
                    pickle.dump(self.reward_his, open(self.reward_his_path, 'wb'), True)
            print('Step: %i/%i,  Episode: %i,  Action: %i,  Episode Reward: %.0f,  Epsilon: %.2f, Loss: %.5f' % (
                step, self.n_steps, episode, action, episode_reward, self.epsilon, loss3))
        self.env.close()

    def get_w(self):
        """
        :return: 获取W值
        """
        return self.sess.graph.get_tensor_by_name('reward/R/kernel:0')

    def __ouput_graph(self):
        """
        输出计算图
        :return:
        """
        directory = "./log"
        try:
            if not os.path.exists(directory):
                os.mkdir(directory)
        except:
            print("Filed to Create result directory: %s" % directory)
            directory = "./"
        self.train_writer = tf.summary.FileWriter(directory, self.sess.graph)

    def store_trainsition(self, s, a, r, d, s_):
        """
        :param s: current state.
        :param a: choiced action.
        :param r: reward.
        :param d: wheathre done.
        :param s_: next state.
        """
        transition = np.hstack((np.reshape(s, [-1]), [a, r, int(d)], np.reshape(s_, [-1])))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choice_action(self, observation, test=False):
        """
        :param observation: current state.
        :param test
        :return: action index.
        """
        if test: self.epsilon = 0.01
        if np.random.uniform() > self.epsilon:
            observation = np.expand_dims(observation, 0)
            encoded = self.sess.run(self.encoded, feed_dict={self.s_p: observation})
            mus_eval = self.sess.run(self.mus_eval, feed_dict={self.fi_eval_p: encoded})
            mus = tf.constant(np.squeeze(np.stack(mus_eval), axis=1))
            Q_tensor = tf.matmul(mus, self.get_w())
            action_index = int(self.sess.run(tf.argmax(Q_tensor))[0])
        else:
            action_index = np.random.randint(0, self.n_actions)
        return action_index

    def __get_model_save_path(self):
        directory = "./model"
        try:
            if not os.path.exists(directory):
                os.mkdir(directory)
        except:
            print("Filed to create result directory.")
            directory = "./"
        self.reward_his_path = os.path.join(directory, "reward_his_dqn.pkl")
        self.model_path = directory
