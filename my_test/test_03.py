#python 对文件，数据库的操作
import sys
import os
import sqlite3
#对文件操作的函数，按行读取
def fileOperate_1():
	fp = open("test_txt.txt","r")
	zops = fp.readlines()
	fp.close()
	i = 1
	print("the zen of Python:")
	for zen in zops:
		print(zen,end="")
		i +=1
#如果对文件打开后不进行操作，可以使用with语句，使操作更方便
#离开with语句后，文件会自动关闭
def fileOperate_2():
	zops=[]
	with open("test_txt.txt") as fp:
		zops = fp.readlines()
		for li in zops:
			print(li,end="")
#为了是程序更具有灵活性，获取文件路径的参数
#可以用sys.argv函树来获取,要先导入sys
def fileOperate_3():
	message = dict()
	studata = list()
	with open(sys.argv[1],encoding='utf-8') as fp:
		alldata = fp.readlines()
	for data in alldata:
		no,name,age,gendle,math,chinese,english = data.rstrip('\n').split(',')
		studata.append(name)
		studata.append(age)
		studata.append(gendle)
		studata.append(math)
		studata.append(chinese)
		studata.append(english)
		message[no] = studata
		print (message,end="\n")


#python 对数据库的操作（SQLite数据库）
#首先要导入sqlite3包
#查看数据库
def databaseOperation():
	conn = sqlite3.connect('score.sqlite')
	consur = conn.execute('select * from score;')
	for row in consur:
		print ('NO{}:{},{},{},{},{}'.format(row[0],row[1],row[2],row[3],row[4],row[5]))
	conn.close()
#插入学生数据
def insertData():
	conn = sqlite3.connect("score.sqlite")
	conn.execute('insert into score values(3,"小明11",11,22,33,45);')
	conn.commit()
	conn.close()

#列出文件夹下所有文件
def sampletree():
	sample_tree = os.walk("my_test")
	for files,subdir,dirname in sample_tree:
		print (files)
		print (subdir)
		print (dirname)

sampletree();

#databaseOperation()