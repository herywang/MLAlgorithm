import numpy as np
import cv2


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
    CC, HH, WW = w.shape[1], w.shape[2], w.shape[3]

    pad = conv_param['pad']
    stride = conv_param['stride']

    x_pad = np.pad(x, ((0, 0), (pad, pad), (pad, pad), (0, 0)), 'constant',
                   constant_values=[[0, 0], [0, 0], [0, 0], [0, 0]])

    Hhat = 1 + (H - HH + 2 * pad) // stride
    What = 1 + (W - WW + 2 * pad) // stride

    out = np.zeros((N, Hhat, What, CC))
    for n in range(N):
        for f in range(CC):
            for i in range(Hhat):
                for j in range(What):
                    kernel = w[:, f, :, :]
                    kernel_reshape = kernel.reshape(kernel.shape[1], kernel.shape[2], kernel.shape[0])
                    out[n, i, j, f] = np.sum(x_pad[n, i * stride: i * stride + HH, j * stride:j * stride + WW, :] \
                                             * kernel_reshape) + b[f]
    cache = (x, w, b, conv_param)
    return out, cache


if __name__ == '__main__':
    image = np.ones([1, 5, 5, 3]) * 2
    cv2.imshow("image", np.asarray(image.reshape([5, 5, 3]), np.uint8))
    w = np.ones([3, 3, 3, 3]) * 3
    b = np.ones([3])
    conv_param = {'pad': 2, 'stride': 1}
    out, cache = conv_forward_naivel(image, w, b, conv_param)
    print(out.shape)
    cv2.imshow("conv", np.asarray(out.reshape([7, 7, 3]), np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(out.reshape([7, 7, 3]))
