import numpy as np
import pandas as pd


def prepare_data():
    # user based collaborative filtering
    data = pd.read_csv('./data.txt', ',', index_col=0, header=None, names=['I1', 'I2', 'I3', 'I4']).values
    means = []
    for line in data:
        sum = 0
        count = 0
        for num in line:
            if not np.isnan(num):
                sum += num
                count += 1
        means.append(sum / count)
    var = []
    i = 0
    for line in data:
        sum = 0
        for num in line:
            if not np.isnan(num):
                sum += (num - means[i])**2
        var.append(sum**0.5)
        i += 1
    corr = np.zeros([data.shape[0], data.shape[0]],dtype=np.float32)

    for i in range(data.shape[0]):
        for j in range(data.shape[0]):
            sum = 0
            for x,y in zip(data[i], data[j]):
                if not np.isnan(x) and not np.isnan(y):
                    sum += (x-means[i])*(y-means[j])
            corr[i,j] = (sum / (var[i] * var[j])) if (var[i] * var[j]) != 0 else 0

    # 估计u1会对i2打多少分
    sum = 0
    for u in range(data.shape[0]):
        if not np.isnan(data[u, 1]):
            sum += (data[u, 1] - means[u]) * corr[0, u]
    result = means[0] + sum / np.sum(np.abs(corr[0]))
    #
    print(result)

    # print(means)
    # print(var)
    # print(corr)


if __name__ == '__main__':
    prepare_data()
