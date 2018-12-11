import tensorflow as tf
import pandas as pd
import numpy as np

def model(user_batch, item_batch, user_num, item_num, dim=5, device='/cpu:0'):
    with tf.device('/cpu:0'):
        with tf.variable_scope('lsi', reuse=True):
            bias_global = tf.get_variable('bias_global', shape=[])

            w_bias_user = tf.get_variable('embd_bias_user', shape=[user_num])