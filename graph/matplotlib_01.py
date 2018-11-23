# -*- coding:utf-8 -*-
#这一个程序主要是几个简单的绘图软件
import matplotlib.pyplot as plt
import csv
def drawLine():
    x=[1,3,5,7,9,11,13,15]
    y=[12.3,12.4,12.8,13.09,14.56,13.67,15.012,14.177]
    plt.plot(x,y,label="First Line",color="g")
    plt.plot([2, 4, 6, 8, 10, 12, 14, 16], [13.3, 13.4, 13.8, 14.09, 15.56, 14.67, 16.012, 15.177], label="Second Line",
            color="r")
    plt.bar(x,y,label="BarData",color="g")
    plt.bar([2,4,6,8,10,12,14,16],[13.3,13.4,13.8,14.09,15.56,14.67,16.012,15.177], label="BarDaRta", color="r")
    plt.legend()#legend函数生成默认图例
    plt.xlabel("次数")
    print("次数")
    plt.ylabel("数量")
    plt.title("2017年10月21日")
    plt.show()
def drawHist():
    population_ages = [22, 55, 62, 45, 21, 22, 34, 42, 42, 4, 99,14,15, 102, 110, 120, 121, 122, 130, 111, 115, 112, 80, 75, 65, 54, 44, 43, 42, 48]
    bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130]
    plt.hist(population_ages, bins, histtype='bar', rwidth=0.9)
    #hist()函数绘制直方图，要传递的参数为hist(总的数据列表，数据区间梯度，直方图histtype = "bar",直方图条宽度rwidth=  )
    plt.xlabel('x')#绘制x轴所代表的变量
    plt.ylabel('y')#绘制y轴所代表的变量
    plt.title('Interesting Graph\nCheck it out')
    plt.legend()
    plt.show()
def drawScatter():
    x=[1,2,3,4,5,6,7,8]
    y=[5,2,4,6,12,11,9,12]
    plt.scatter(x,y,color="r",s=100,marker="+")
    #x,y是坐标，color是set color，'s' is set the size of point,'marker' is used to set the style of the point
    plt.title("Scatter Photo")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim(0,10)
    plt.ylim(0,15)
    plt.show()
def drawStackPlot():
    days =   [1.0,2.0,3.0,4.0,5.0,6.0,7.0]
    sleeping=[6.0,6.5,6.2,6.9,7.0,7.1,6.2]
    eating=  [1.0,1.2,0.9,0.8,1.0,1.1,1.2]
    learing= [9.2,9.3,9.1,7.0,5.1,6.8,8.1]
    working= [2.0,2.4,4.0,3.2,1.9,0.8,0.5]
    playing= [5.8,4.6,3.8,6.1,9.0,8.2,8.0]
    plt.plot([],[],label="sleeping",color='y',linewidth="5")
    plt.plot([],[],label="eating", color='r', linewidth="5")
    plt.plot([],[],label="study", color='m', linewidth="5")
    plt.plot([],[],label="working", color='k', linewidth="5")
    plt.plot([],[],label="plaing", color='c', linewidth="5")
    plt.stackplot(days, sleeping,eating,learing,working,playing,colors=['y','r','m','k','c'])
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()
def drawPie():
    slices=[2,7,7,12]#数据
    activities=['sleeping','eating','working','playing']#设置标签属性
    colors=['c','m','r','b']

    plt.pie(slices,
            labels=activities,
            colors=colors,
            startangle=90,#开始绘制饼图的角度,按照逆时针绘制
            shadow=True,#阴影
            explode=(0.1, 0.0, 0.2, 0),#将第一个切片拉出0.1，
            autopct='%d%%')#将百分比放上
    plt.title('Interesting Graph\nCheck it out')
    plt.show()
def drawbyfilefata():
    x=[]
    y=[]
    with open("data.csv","r") as csvfile:
        plots = csv.reader(csvfile,delimiter=",")
        for plot in plots:
            x.append(float(plot[0]))
            y.append(float(plot[1]))
    plt.plot(x,y,label="csv File data",color="r")
    plt.title("CSV FILE DATA")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()

#drawLine() 绘制直线图&绘制条形图
#drawHist() 绘制直方图
drawScatter()#绘制散点图
#drawStackPlot()#绘制堆叠图
#drawPie()
#drawbyfilefata()