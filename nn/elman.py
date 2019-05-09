import tensorflow as tf
import numpy as np

INPUT_UNITS = 4
HIDDEN_UNITS = 9
UNDER_TAKE_UNITS = HIDDEN_UNITS
OUTPUT_UNITS = 1
'''
    Elman神经网络手写实现.
'''

def build_model():

    with tf.name_scope("input"):
        x = tf.placeholder(dtype=tf.float32, name="input", shape=[None, INPUT_UNITS])
        y = tf.placeholder(dtype=tf.float32, name='output', shape=[None, OUTPUT_UNITS])
        undertake = tf.placeholder(dtype=tf.float32, name="undertake", shape=[None, HIDDEN_UNITS])  # 承接层输入 -- 初始值为0

    with tf.name_scope("input-layer"):
        w1 = tf.Variable(tf.random.truncated_normal((INPUT_UNITS, HIDDEN_UNITS)),
                         name="input-weights")  # 输入层权重
        b1 = tf.Variable(tf.random.truncated_normal((1, HIDDEN_UNITS)))
        x_ck = tf.nn.sigmoid(tf.matmul(x, w1) + b1)

    with tf.name_scope("undertake-layer"):
        w2 = tf.Variable(tf.random.truncated_normal((HIDDEN_UNITS, HIDDEN_UNITS)),
                         name="undertake-weights")  # 承接层权重
        b2 = tf.Variable(tf.random.truncated_normal((1, HIDDEN_UNITS)))
        u_k_1 = tf.nn.sigmoid(tf.matmul(undertake, w2) + b2)

    with tf.name_scope("hidden-layer"):
        hidden = tf.add(x_ck, u_k_1, 'add')

    with tf.name_scope("output-layer"):
        w3 = tf.Variable(tf.random.truncated_normal((HIDDEN_UNITS, OUTPUT_UNITS)),
                         name="output-weights")
        b3 = tf.Variable(tf.random.truncated_normal((1, OUTPUT_UNITS)))
        x_k = tf.nn.relu(hidden * w3 + b3)

    with tf.name_scope("loss-value"):
        RMSE = tf.sqrt(tf.reduce_mean(tf.square(x_k - y)))

    tf.summary.scalar("RMSE-loss", RMSE)

    opt = tf.train.AdadeltaOptimizer(0.001).minimize(RMSE)

    merged = tf.summary.merge_all()

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    train_writer = tf.summary.FileWriter("./log", sess.graph)
    undertake_k_1 = np.zeros((1, HIDDEN_UNITS))

    input_x, input_y = get_input_data()
    batches = input_x.shape[0]
    for i in range(2500):
        for j in range(batches):
            _, undertake_k_1 = sess.run([opt, x_k], feed_dict={x: input_x, y: input_y, undertake: undertake_k_1})
        if i % 10 == 0:
            summary = sess.run(merged, feed_dict={x: input_x, y: input_y, undertake: undertake_k_1})
            train_writer.add_summary(summary)


def get_input_data():

    x = np.random.randint(0, 10, [1000, 4], dtype=np.float32)
    y = np.random.randint(10,100,[1000], dtype=np.float32)

    return x, y

build_model()