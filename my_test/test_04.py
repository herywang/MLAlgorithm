# coding=utf-8
import pandas as pd
import os
import requests

def data_01():
    count = int(input("请输入累加元素个数："))
    my_data = list()
    for num in range(0,count):
        my_num = float(input("请输入第"+str(num+1)+"个数据："))
        my_data.append(my_num)
    print(my_data)
    ss = 0.0
    for num in my_data:
        ss += num
    di = ss/(len(my_data))
    print(di)
def ML_test():
    PATH = r'/'