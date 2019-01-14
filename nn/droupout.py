# The implemention of dropout function.
import numpy as np


def dropout_forward(x, dropout_paramm):
    """
    Inputs:
     -x: input value
     -dropout_param: this is an dictionary's type data, use these keys:
        -p: dropout activity parameter, the probabilistic of each neurons.
        -mode: 'test' or 'train',
            train: the activation probability 'p' is used to perform the
                    'and' operation with neuron.
            test: remove the activation probability 'p' and return only the input
                    value.
        -seed: the seed of random
    """
    p, mode = dropout_paramm['p'], dropout_paramm['mode']
    retrain_p = 1 - p
    if 'seed' in dropout_paramm:
        np.random.seed(dropout_paramm['seed'])
    mask = None
    out = None
    if mode == 'train':
        mask = np.random.binomial(1, retrain_p, x.shape)
        out = x*mask
    elif mode == 'test':
        out = x
    cache = (dropout_paramm, mask)
    out = out.astype(x.dtype, copy=False)
    return out, cache