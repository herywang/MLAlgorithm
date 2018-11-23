#python的分支语句
#coding=utf-8
#test_01.py (Python 3 version)
import re
def try_expect( a ):
    while a:
        try:
            a = int(input("请输入您的年龄："))
            break
        except:
            print("请输入您的年龄:")
    if a<15:
        print("您太小了!")
    else:
        print("合格")

def draw_bar(n,a="*"):
    for i in range(1,n):
        print(a,end="\t")

def open_file(filename):
    fp=open(filename,"r")
    read = fp.read()
    new_read = re.sub("[^a-zA-Z\s]","",read)
    words = new_read.split()
    words_counts = {}
    for word in words:
        if word.upper() in words_counts:
            words_counts[word.upper()] = words_counts[word.upper()]+1
        else:
            words_counts[word.upper()] = 1
    key_list = list(words_counts.keys())
    key_list.sort()

    for key in key_list:
        if words_counts[key] >1:
            print("{}:{}".format(key,words_counts[key]))
