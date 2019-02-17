import tensorflow as tf
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


def prepare_data():
    original_data = pd.read_csv("./heart.csv")
    data = original_data.values
    data[:, 0] = data_normalization(data[:, 0])
    data[:, 3] = data_normalization(data[:, 3])
    data[:, 4] = data_normalization(data[:, 4])
    data[:, 7] = data_normalization(data[:, 7])

    X_ = data[:, :-1]
    y_ = data[:, -1]
    x_train, x_test, y_train, y_test = train_test_split(X_, y_, test_size=0.2, shuffle=True, random_state=0)
    return x_train, x_test, y_train, y_test


def data_normalization(data, method='max-min'):
    if method == 'max-min':
        max_value, min_value = max(data), min(data)
        data = (data - np.repeat(min_value, data.shape[0])) / (max_value - min_value)
        return data

    elif method == 'z-zero':
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        return (data - np.repeat(mean, data.shape[0])) / std


def build_model(x_train, x_test, y_train, y_test):

    graph = tf.Graph()
    with graph.as_default():
        with graph.name_scope("input"):
            X_ = tf.placeholder(dtype=tf.float32, shape=[None, 13], name="X")
            y = tf.placeholder(dtype=tf.float32, shape=[None], name="Y")

        with graph.name_scope("layer-1"):
            test = tf.get_variable("test_variable", initializer=tf.random_normal([19, 64]))
            w1 = tf.Variable(tf.random_normal([13, 64]), name="w1")
            b1 = tf.Variable(tf.random_normal([64]), name="b1")
            layer1 = tf.matmul(X_, w1) + b1
            layer1 = tf.nn.sigmoid(layer1, "SigmoidAct")
            res1 = tf.nn.dropout(layer1, 0.3)

        with graph.name_scope("layer-2"):
            w2 = tf.Variable(tf.random_normal([64, 64]), name="w2")
            b2 = tf.Variable(tf.random_normal([64]), name="b2")
            layer2 = tf.matmul(res1, w2) + b2
            layer2 = tf.nn.sigmoid(layer2, "SigmoidAct")
            res2 = tf.nn.dropout(layer2, 0.3)

        with graph.name_scope("layer-3"):
            w3 = tf.Variable(tf.random_normal([64, 1]), name="w3")
            b3 = tf.Variable(tf.random_normal([1]), name="b3")
            layer3 = tf.matmul(res2, w3) + b3
            res3 = tf.nn.sigmoid(layer3)
        with graph.name_scope("loss"):
            loss = tf.reduce_mean(tf.square(res3 - y),name="loss-value")
        tf.summary.scalar("loss", loss)
        merged = tf.summary.merge_all()
        train = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
        with tf.Session(graph=graph) as sess:
            train_writer = tf.summary.FileWriter("./train", sess.graph)
            tf.global_variables_initializer().run(session=sess)

            for i in range(100000):
                _, merged_ = sess.run([train, merged], feed_dict={X_: x_train, y: y_train})
                if i % 100 == 0:
                    train_writer.add_summary(merged_, i)
                    print("setp: {}, loss value is {:.5f}:".format(i, sess.run(loss, feed_dict={X_: x_train, y: y_train})))
            train_writer.close()

if __name__ == '__main__':
    x_train, x_test, y_train, y_test = prepare_data()
    build_model(x_train, x_test, y_train, y_test)



