import numpy as np


def conv_forward_naivel(x, w, b, conv_param):
    """
    Input:
        - x: 4-dim image data: (N, H, W, C), which represent: image's number, image's width, image's height, image's channel.
        - w: 4-dim convolution kernel: (F, C, HH, WW), which represent: input channel, output channel, kernel height,
            kernel width.
        - b: bias.
        - conv_param: dictionary type. key is :
            - stride: the number of spans of the data to be convolved.
            - pad: the number of zero padding of input data.
    Returns: tuple
        - out: output data(N, F, H1, W1)
            H1 = 1 + (H - HH + 2P)/stride
            W1 = 1 + (W - WW + 2P)/stride
        - cache: (x, w, b, conv_param)
    """
    N, H, W, C = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
    C, HH, WW = w.shape[1], w.shape[2], w.shape[3]

    pad = conv_param['pad']
    stride = conv_param['stride']

    x_pad = np.pad(x, ((0, 0), (pad, pad), (pad, pad), (0, 0)), 'constant',
                   constant_values=[[0, 0], [0, 0], [0, 0], [0, 0]])

    Hhat = 1 + (H - HH + 2 * pad) // stride
    What = 1 + (W - WW + 2 * pad) // stride

    out = np.zeros([N, Hhat, What, w[1]])
    for n in range(N):
        for f in range(C):
            for i in range(Hhat):
                for j in range(What):
                    kernel = w[:, f, :, :]
                    kernel_reshape = kernel.reshape(kernel[1], kernel[2], kernel[0])
                    out[n, i, j, f] = x_pad[n, i * stride: i * stride + HH, j * stride:j * stride + WW, :]\
                                      * kernel_reshape + b[f]
    cache = (x, w, b, conv_param)
    return out, cache
