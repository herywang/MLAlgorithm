# -*- coding utf-8 -*-
import numpy as np
from collections import defaultdict
dataset_filename = "affinity_dataset.txt"
x = np.loadtxt(dataset_filename)
features = ["bread", "milk", "cheese", "apples", "bananas"]
valid_rules=defaultdict(int)
invalid_rules=defaultdict(int)
num_occurances=defaultdict(int)
def demo2():
    for sample in x:
        for premist in range(5):
            if sample[premist]==0:continue
            num_occurances[premist]+=1
            for conclusion in range(5):
                if conclusion==premist:continue
                if sample[conclusion] == 1:
                    valid_rules[(premist,conclusion)] +=1
                else:
                    invalid_rules[(premist,conclusion)] +=1
    support = valid_rules
    confident = defaultdict(float)
    for premist,conclusion in support.keys():
        confident[(premist,conclusion)] = support[(premist,conclusion)]/num_occurances[premist]
    for premist,conclusion in confident:
        premist_name=features[premist]
        conclusion_name=features[conclusion]
        print("规则：如果一个人购买了{0},他还会购买{1}的概率为：{2:.4f}".format(premist_name,conclusion_name,confident[(premist,conclusion)]))
        print("支持度为：{0}".format(support[(premist,conclusion)]))
        print(("-"*30))

def test():
    dd=defaultdict(int)
    dd[(1,3)]+=12
    dd[(2,4)]+=11
    for k,v in dd:
        print(v)
demo2()
