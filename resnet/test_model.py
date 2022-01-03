import numpy as np
import torch
import torch.nn as nn


def test_padding():
    image = torch.Tensor(np.reshape(np.random.rand(3 * 150 * 150), [1, 3, 150, 150]))
    conv = nn.Conv2d(3, 6, (5, 1), (1, 1), padding=(2, 0))
    flat = nn.Flatten()
    print(image.shape)
    conv_image = conv(image)
    print(conv_image.shape)
    flatten = flat(conv_image)
    print(flatten.shape)


def test_identity():
    image = torch.FloatTensor(np.reshape(np.random.rand(3 * 3 * 3), [1, 3, 3, 3]))
    short_cut = nn.Identity()
    print(image)
    print(short_cut(image))


if __name__ == '__main__':
    # test_padding()
    test_identity()
