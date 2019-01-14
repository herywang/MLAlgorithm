<<<<<<< HEAD
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
        cost = -tf.reduce_mean(tf.add(tf.multiply(self.y, tf.log(r_y_)),\
            tf.multiply(tf.subtract(1.0, self.y), tf.log(r_1_y))))
        self.J = cost + self.regcoef * tf.nn.l2_loss([self.W1])
        self.train_op = tf.train.AdamOptimizer(0.0001).minimize(self.J)

    def train(self, model=TRAIN_MODEL_NEW, ckpt_file='work/dae.ckpt'):
        pass
=======
#实现仿射传播
from __future__ import absolute_import, division, print_function

import numpy as np

class NeuralNetwork:
    def __init__(self):
        pass


    @staticmethod
    def affine_forward(x, w, b):
        """
        :param x: shape:[N, D]
        :param w: shape:[D, M]
        :param b: shape:[M]
        :return: out: shape[N, M], cache: shape(x, w, b)
        """
        out = np.dot(x, w) + b
        cache = (x, w, b)
        return out, cache
>>>>>>> b986d7aaae6c07f2b0775cddd148bd0b13e7e78b
