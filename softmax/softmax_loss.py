import numpy as np
def softmax_loss_naive(w, x, y, reg):
    """
    使用显示循环版本计算softmax损失函数
    N:数据个数， D：数据维度， C:数据类别个数
    inputs:
    :param w: shape:[D, C], 分类器权重参数
    :param x: shape:[N, D] 训练数据
    :param y: shape:[N, ]数据标记
    :param reg: 惩罚项系数
    :return:二元组：loss:数据损失值， dw：权重w对应的梯度，其形状和w相同
    """
    loss = 0.0
    dw = np.zeros_like(W)
    #############################################################################
    #  任务：使用显式循环实现softmax损失值loss及相应的梯度dW 。                    #
    #  温馨提示： 如果不慎,将很容易造成数值上溢。别忘了正则化哟。                   #
    #############################################################################
    num_train = x.shape[0]
    num_class = w.shape[1]
    for i in range(num_train):
        s = x[i].dot(w)
        score = s - np.max(s)
        score_E = np.exp(score)
        z = np.sum(score_E)
        score_y_o = score_E[y[i]]
        loss += -np.log(score_y_o / Z)
        for j in range(num_class):
            if j == y[i]:
                dw[:, j] += -(1-score_E[j]/z) * x[i]
            else:
                dw[:, j] += x[i] * score_E[j]/z
    loss = loss/num_train + 0.5*reg*np.sum(w*w)
    dw = dw / num_train + reg*w

    return loss, dw