# -*- coding utf-8 -*-
from os import  listdir
#将图片文件转换为向量
def img2vector(filename):
   with open(filename) as fobj:
       arr = fobj.readlines()
   vec, demension = [], len(arr)
   for i in range(demension):
        line = arr[i].strip()
        for j in range(demension):
             vec.append(int(line[j]))
   return vec
#读取训练数据
def createDataset(dir):
    dataset, labels = [], []
    files = listdir(dir)
    for filename in files:
        label = int(filename[0])
        labels.append(label)
        dataset.append(img2vector(dir + '/' + filename))

    return dataset, labels

#计算谷本系数
def tanimoto(vec1, vec2):
    c1, c2, c3 = 0, 0, 0
    for i in range(len(vec1)):
        if vec1[i] == 1: c1 += 1
        if vec2[i] == 1: c2 += 1
        if vec1[i] == 1 and vec2[i] == 1: c3 += 1

    return c3 / (c1 + c2 - c3)

def classify(dataset, labels, testData, k=20):
    distances = []

    for i in range(len(labels)):
        d = tanimoto(dataset[i], testData)
        distances.append((d, labels[i]))

    distances.sort(reverse=True)
    #key  label,  value   count of the label
    klabelDict = {}
    for i in range(k):
        klabelDict.setdefault(distances[i][1], 0)
        klabelDict[distances[i][1]] += 1 / k

    #按value降序排序
    predDict = sorted(klabelDict.items(), key=lambda item: item[1], reverse=True)
    return predDict
dataset, labels = createDataset('trainingDigits')
testData = img2vector('testDigits/8_19.txt')
print(classify(dataset, labels, testData))