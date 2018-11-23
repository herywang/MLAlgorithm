# -*- coding utf-8 -*-
import numpy as np
data = np.random.random([1,8])
print(data)
data.shape = (2,4)
print(data)

a= np.arange(6,10)
b= np.arange(5,9)
d= np.arange(11,15)
e = a*b
c= a+b
print(c)
print(e)



