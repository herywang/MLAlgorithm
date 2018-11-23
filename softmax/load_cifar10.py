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

if __name__ == '__main__':
    load_CIFAR_BATCH(r'D:\B-WorkSpace\PycharmProjects\DATASETS\cifar-10-batches-py\data_batch_1')
