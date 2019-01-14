# 堆叠自动编码机
# Stacking automatic encoder

from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

class NeuralNetwork:
    TRAIN_MODEL_NEW=1
    TRAIN_MODEL_CONTINUE=2

    def __init__(self):
        self.hidden_size=1024
        self.n = 784
        self.regcoef = 5e-4
        self.tf_graph = tf.Graph()
        self.num_epochs = 200
        self.corr_frac = 0.1 # 加入噪音比例
        self.batch_size = 128
        self.X = None
        self.y = None
        self.y_ = None
        self.keep_prob = None
        self.W1 = None
        self.b2 = None
        self.b3 = None
        self.a2 = None
        self.J = None
        self.train_op = None


    def load_dataset(self):
        mnist = input_data.read_data_sets('/home/wangheng/workspace/PycharmProjects/MNIST_data', one_hot=True)
        X_train = mnist.train.images
        y_train = mnist.train.labels
        X_test = mnist.test.images
        y_test = mnist.test.labels
        X_validation = mnist.validation.images
        y_validation = mnist.validation.labels
        return X_train, y_train, X_test, y_test, X_validation, y_validation

    def add_noise(self, sess, X, corr_frac):
        X_prime = X.copy()
        rand = tf.random_uniform(X.shape)
        X_prime[sess.run(tf.nn.relu(tf.sign(corr_frac - rand))).astype(np.bool)] = 0
        return X_prime

    def get_mini_batchs(self, X, batch_size):
        X = np.array(X)
        length = X.shape[0]
        for i in range(0, length, batch_size):
            if i+batch_size < length:
                yield X[i: i+batch_size]
            else:
                yield X[i: ]

    def build_model(self):
        print("Building auto-denoising Autoencoder Model v 1.0")
        print("Beginning to build the model")
        self.X = tf.placeholder(dtype=tf.float32, shape=[None, self.n])
        self.y = tf.placeholder(dtype=tf.float32, shape=[None, self.n])
        self.keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')
        self.W1 = tf.Variable(initial_value=tf.truncated_normal(shape=[self.n, self.hidden_size], mean=0.1, stddev=0.1),\
            name='w1')
        self.b2 = tf.Variable(tf.constant(0.001, shape=[self.hidden_size]), name='b2')
        self.b3 = tf.Variable(tf.constant(0.001, shape=[self.n]), name='b3')
        with tf.name_scope('encoder'):
            z2 = tf.matmul(self.X, self.W1) + self.b2
            self.a2 = tf.nn.tanh(z2)
        with tf.name_scope('decoder'):
            z3 = tf.matmul(self.a2, tf.transpose(self.W1)) + self.b3
            a3 = tf.nn.tanh(z3)
        self.y_ = a3
        r_y_ = tf.clip_by_value(self.y_, 1e-10, float('inf'))
        r_1_y = tf.clip_by_value(1-self.y_, 1e-10, float('inf'))
        cost = tf.reduce_mean(tf.add(tf.multiply(self.y, tf.log(r_y_)),\
            tf.multiply(tf.subtract(1.0, self.y), tf.log(r_1_y))))
        self.J = cost + self.regcoef * tf.nn.l2_loss([self.W1])
        self.train_op = tf.train.AdamOptimizer(0.0001).minimize(self.J)

    def train(self, model=TRAIN_MODEL_NEW, ckpt_file='work/dae.ckpt'):
        X_train, y_train, X_test, y_test, X_validation, y_validation = self.load_dataset()
        with self.tf_graph.as_default():
            self.build_model()
            saver = tf.train.Saver()
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                for epoch in range(self.num_epochs):
                    X_train_prime = self.add_noise(sess, X_train, self.corr_frac)
                    shuff = list(zip(X_train, X_train_prime))
                    np.random.shuffle(shuff)
                    batches = [_ for _ in self.get_mini_batchs(shuff, self.batch_size)]
                    batch_idx = 0
                    for batch in batches:
                        X_batch_raw, X_prime_batch_raw = zip(*batch)
                        X_batch = np.array(X_batch_raw).astype(np.float32)
                        X_prime_batch = np.array(X_prime_batch_raw).astype(np.float32)
                        batch_idx += 1

                        opv, loss = sess.run([self.train_op, self.J], feed_dict={self.X: X_prime_batch, self.y: X_batch})
                        if batch_idx % 100 == 0:
                            print('eposh{0}_batch{1}: {2}'.format(epoch, batch_idx, loss))
                            saver.save(sess, ckpt_file)
    
    def run(self, ckpt_file='work/dae.ckpt'):
        img_file = 'datasets/test'

if __name__ == '__main__':
    nn = NeuralNetwork()
    nn.train()