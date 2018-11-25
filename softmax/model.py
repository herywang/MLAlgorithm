from __future__ import absolute_import, print_function, division

import numpy as np
from .softmax_loss import Softmax

class Model:

    def __init__(self):
        self.input = None
        self.target = None
        self.w = None
        self.loss_history = None

    def train(self, x, y, learning_rate=1e-3, reg=1e-5, num_iter=100, batch_size=10, verbos=0):
        """
        Training model.
        :param x: input value, shape:(N, D), N: the values of sample, D:the sample's dimension.
        :param y: target value, shape:(N,)
        :param learning_rate: learning rate, default: 1e-3
        :param reg: regularization coefficient.
        :param num_iter: train numbers.
        :param batch_size: train batch's size.
        :param verbos: if verbos=0, training process can not visualized.
        :return: loss history, type: list
        """
        num_train, num_dim = np.asarray(x, dtype=np.int).shape
        num_class = int(np.max(y) + 1)
        if self.w is None:
            self.w = np.random.randn(num_dim, num_class)
        loss_history = []
        for i in range(num_iter):
            indices = np.random.choice(num_train, batch_size,False)
            x_batch = x[indices, :]
            y_batch = y[indices]
            loss, gradient = Softmax.softmax_loss_matrix(self.w, x_batch, y_batch, reg)
            loss_history.append(loss)
            self.w = self.w - learning_rate * gradient
            if verbos>0 and i % 100 == 0:
                print("iter: %d\t loss:%f"%(i, loss))
        self.loss_history = loss_history
        return loss_history
