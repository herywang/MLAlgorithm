import matplotlib.pyplot as pt
w=[1,2,5,2,7]
x=[12,23,11,23,11]
y=[1,3,8,4,0]
z=[17,11,12,19,10]

pt.plot(x,y,lw=2,label='Marry')
pt.plot(w,z,lw=2,label='Tom')
pt.xlabel('month')
pt.ylabel('dollars(million)')
pt.legend()
pt.title('Proper test')
pt.show()