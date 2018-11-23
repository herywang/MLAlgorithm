# -*- coding utf-8 -*-
import pandas as pd
import numpy as np



def dataframe1():
    s = pd.Series([-12, -34, 90, 34], index=['e', 'f', 'g', 'h'])
    print(s)
    print(s.values)
    print(s.index)
    print(s[2])  # 输出第二个元素
    print(s['g'])  # 根据索引找到g多对应的元素的值
    print(s[['e', 'h']])  # 输出e ,h多对应的值，包括索引，以及values
    s['f'] = 55  # 为元素复制
    print(s)

    data1 = np.arange(4, 10)
    ss = pd.Series(data1)
    print(ss)
    data2 = np.arange(3, 7)  # 产生3到7的一维数组
    sss = pd.Series(data2)  # a将数组的值作为series索引的值
    print(sss)
    print(sss[3])
    print(sss[sss > 4])  # 筛选元素
    print(sss / 2)  # Series对象的运算
    print(np.sin(sss))  # series
    print(sss.unique())
    # Series 用作字典
    mydict = {'red': 12, 'blue': 14, 'yello': 90, 'green': -78}
    print(mydict)
    print(pd.Series(mydict))
    colors = ['red', 'blue', 'yello', 'green']
    print(pd.Series(mydict, index=colors))  # 用字典的值来填充collors键
    print("=" * 30)
    print(pd.Series(mydict, index=colors) + pd.Series(mydict))

    # ]]]]]]======dataFrame对象========[[[[[[[
    # dataFrame对象要传递给一个dict给DataFrame（）构造函数以每一列的的名称作为键，每个键都有一个数组作为值
    mydict_1 = {'color': ['red', 'green', 'yellow', 'black'], 'object': ['blood', 'tree', 'video', 'pen'],
                'value': [12, 34.8, 23.9, -11.98]}
    frame = pd.DataFrame(mydict_1)  # 构建一个frameData对象
    print(frame)
    # 指定frameData对象的显示列
    frame_1 = pd.DataFrame(mydict_1, columns=['color', 'value'])
    print(frame_1)
    # 指定dataFrame的索引
    frame_2 = pd.DataFrame(mydict_1, index=['one', 'two', 'three', 'four'], columns=['color', 'value'])
    print(frame_2)
    # ========选取元素======
    print(frame_2.values)
    # ========选取index======
    print(frame_2.keys())
    # 选取某一个索引的值的两种方法
    print(frame_2['value'])
    print(frame_2.color)
    # ======获取某一行的元素,通过ix属性以及行的索引获取
    print(frame_2.ix['two'])
    # 通过ix获取多个索引值所对应得值.注意：要传递多个索引时，索引必须要用[]括起来
    print(frame_2.ix[['one', 'three']])
    # 选取DATa Frame 的一部分：
    print(frame[0:2])
    # 选取dataFrame的一个元素,第一个参数是  列 ，，第二个是  行
    print(frame['color'][2])
    # 添加一个新列
    frame['new'] = 13
    print(frame)

    # =================================================================
    # 更新某一列的数据,首先要构建一个Series对象,然后利用dataFrame对象设置
    ser = pd.Series(np.arange(4))
    frame['new'] = ser
    print(frame, '\n')
    # 修改dataFrame单个元素
    frame['new'][1] = 12
    print(frame)
    # 用isin()判断是否为Series对象中的数据
    print(frame[frame.isin(['red', 12])])
    # 删除一整列
    del frame['new']
    print(frame)
    # dataFrame转置
    print(frame.T, '\n')
    # Series更换索引
    s.reindex([1, 2, 3, 4])
    print(s)
    # ==============================计算两个序列的协方差，以及相关系数=========
    s = pd.Series([-12, -34, 90, 34], index=['e', 'f', 'g', 'h'])
    print(s)
    print(np.arange(5, 15))
    seq1 = pd.Series(np.array([3.45, 1.67, 5.56, 7.89, 9.01]), index=np.arange(1, 6))
    seq2 = pd.Series(np.arange(5, 10), index=np.arange(1, 6))
    print(seq1, '\n', seq2, '\n')
    print("协方差：", '\t', seq1.cov(seq2))
    print("相关系数：", seq1.corr(seq2))
    frame = pd.DataFrame(
        [[1, 4, 5, 6], [8.9, 12.6, 19.98, 33.87], [23.11, 34.56, 22.45, 19.987], [-23.1, 90.7, 34.6, 89.4]],
        index=['a', 'b', 'c', 'd'], columns=['d1', 'd2', 'd3', 'd4'])







