# -*- coding: utf-8 -*-
import pickle as p
import numpy as np
import os


def load_CIFAR_BATCH(filename):
    """load one batch file"""
    with open(filename, 'rb') as f:
        datadict = p.load(f, encoding='latin1')
        x = datadict['data']
        y = datadict['labels']
        x = x.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype(np.float32)
        y = np.asarray(y, dtype=np.float32)
        return x, y


def load_cifar10(path):
    xs = []
    ys = []
    for b in range(1, 6):
        f = os.path.join(path, 'data_batch_%d' % (b))
        x, y = load_CIFAR_BATCH(f)
        xs.append(x)
        ys.append(y)
    xs = np.asarray(xs)
    xt = np.concatenate(xs)
    yt = np.concatenate(ys)
    del xs, ys
    xte, yte = load_CIFAR_BATCH(os.path.join(path, 'test_batch'))
    return xt, yt, xte, yte


def get_normalization_cifar10_data(path):
    x_train, y_train, x_test, y_test = load_cifar10(path)
    X_train = x_train.reshape([x_train.shape[0], -1])
    Y_train = y_train.reshape([y_train.shape[0], -1])
    X_test = x_test.reshape([x_test.shape[0], -1])
    Y_test = y_test.reshape([y_test.shape[0], -1])
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    X_train  = (X_train - mean)/std
    X_test = (X_test - mean)/std
    return X_train, Y_train, X_test, Y_test