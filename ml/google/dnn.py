import tensorflow as tf


def build_simple_network(X, is_plane):
    Y1 = tf.layers.dense(X, 200, activation=tf.nn.relu)
    Y2 = tf.layers.dense(Y1, 20, activation=tf.nn.relu)
    Ylogits = tf.layers.dense(Y2, 2)
    loss = tf.losses.softmax_cross_entropy(tf.one_hot(is_plane, 2), Ylogits)
    train_op = tf.train.AdamOptimizer(0.001).minimize(loss)

def build_policy_network():
    observations = tf.placeholder(shape=[None, 80*80])
    actions = tf.placeholder(shape=[None])
    rewards = tf.placeholder(shape=[None])
    # model
    Y = tf.layers.dense(observations, 20, activation=tf.nn.relu)
    YLogits = tf.layers.dense(Y, 3)
    sample_op = tf.multinomial(logits=YLogits, num_samples=1)
