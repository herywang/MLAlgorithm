# Deep successor representation.
import tensorflow._api.v2.compat.v1 as tf

tf.disable_v2_behavior()
tf.disable_eager_execution()


class DSRBrain:
    def __init__(self,action_num, sess: tf.Session, output_graph=False):

        self.action_num = action_num
        self.s_t = tf.placeholder(tf.float32, [None, 40, 40, 1], 's_t')
        if sess is None:
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())
        self.sess = sess
        if output_graph:
            tf.summary.FileWriter("./dsr_log/", self.sess.graph)

    def build_network_architecture(self):
        with tf.variable_scope("f"):
            with tf.variable_scope("conv_1"):
                w1 = tf.get_variable('w1', [8, 8, 1, 32], tf.float32, tf.truncated_normal_initializer())
                b1 = tf.get_variable('b1', [32], tf.float32, tf.truncated_normal_initializer())
                l1 = tf.nn.relu(tf.add(tf.nn.conv2d(self.s_t, w1, [1, 2, 2, 1], 'SAME'), b1))
            with tf.variable_scope('conv_2'):
                w2 = tf.get_variable("w2", [4, 4, 32, 64], tf.float32, tf.truncated_normal_initializer())
                b2 = tf.get_variable('b2', [64], tf.float32, tf.truncated_normal_initializer())
                l2 = tf.nn.relu(tf.add(tf.nn.conv2d(l1, w2, [1, 1, 1, 1], 'SAME'), b2))
            with tf.variable_scope('conv_3'):
                w3 = tf.get_variable('w3', [3, 3, 64, 64], tf.float32, tf.truncated_normal_initializer())
                b3 = tf.get_variable('b2', [64], tf.float32, tf.truncated_normal_initializer())
                l3 = tf.nn.relu(tf.add(tf.nn.conv2d(l2, w3, [1, 1, 1, 1], 'SAME'), b3))
            with tf.variable_scope('fc1'):
                flatten = tf.reshape(l3, [-1, 64 * 3 * 3])
                w_r = tf.get_variable('w_r', [64 * 3 * 3, 512], tf.float32, tf.truncated_normal_initializer())
                b_r = tf.get_variable('b_r', [512], tf.float32, tf.truncated_normal_initializer())
                fai_s = tf.nn.relu(tf.add(tf.matmul(flatten, w_r), b_r))
        with tf.variable_scope('R(s)'):
            w = tf.get_variable('w', [])
