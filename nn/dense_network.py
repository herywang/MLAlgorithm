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
