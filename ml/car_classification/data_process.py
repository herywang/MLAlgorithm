# vehicle classification.
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
pd.set_option('display.max_columns', None)

def load_data():
    col_names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
    data = pd.read_csv('./car.csv', names=col_names)
    return data

def convert2onehot(data):
    return pd.get_dummies(data, prefix=data.columns).values.astype(np.float32)

def model():
    data = load_data()
    np_data = convert2onehot(data)
    np.random.shuffle(np_data)
    sep = int(0.7 * np_data.shape[0])
    train_data = np_data[:sep, :]
    test_data = np_data[sep:, :]

    tf_input = tf.placeholder(tf.float32, shape=[None, train_data.shape[1]], name='input')
    tfx = tf_input[:, :21]
    tfy = tf_input[:, 21:]

    l1 = tf.layers.dense(tfx, 28, activation=tf.nn.relu, name='l1')
    l2 = tf.layers.dense(l1, 128, activation=tf.nn.sigmoid, name='l2')
    l3 = tf.layers.dense(l2, 4, name='l3')
    prediction = tf.nn.softmax(l3, name='pred')

    loss = tf.losses.softmax_cross_entropy(onehot_labels=tfy, logits=l3)
    accuracy = tf.metrics.accuracy(labels=tf.arg_max(tfy, 1), predictions=tf.arg_max(prediction, 1))[1]
    opt = tf.train.GradientDescentOptimizer(0.1)
    train_op = opt.minimize(loss)
    sess=  tf.Session()
    sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))

    plt.ion()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    accuracies, steps = [], []

    for t in range(40000):
        batch_index = np.random.randint(len(train_data), size=32)
        sess.run(train_op, feed_dict={tf_input: train_data[batch_index]})
        if t % 50 == 0:
            acc_, pred_, loss_ = sess.run([accuracy, prediction, loss], feed_dict={tf_input: test_data})
            print('step: %i' % t, '|Accuracy is %.2f'%acc_, '|loss is: %.2f'%loss_)
            accuracies.append(acc_)
            steps.append(t)

            # visualize testing
            ax1.cla()
            for c in range(4):
                bp = ax1.bar(c + 0.1, height=sum((np.argmax(pred_, axis=1) == c)), width=0.2, color='red')
                bt = ax1.bar(c - 0.1, height=sum((np.argmax(test_data[:, 21:], axis=1) == c)), width=0.2, color='blue')
            ax1.set_xticks(range(4), ["accepted", "good", "unaccepted", "very good"])
            ax1.legend(handles=[bp, bt], labels=["prediction", "target"])
            ax1.set_ylim((0, 400))
            ax2.cla()
            ax2.plot(steps, accuracies, label="accuracy")
            ax2.set_ylim(ymax=1)
            ax2.set_ylabel("accuracy")
            plt.pause(0.01)
    plt.ioff()
    plt.show()

if __name__ == '__main__':
    model()