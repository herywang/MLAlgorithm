# -*- coding utf-8 -*-
import numpy as np
def test01():
   b=np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,17, 18, 19, 20, 21, 22, 23])
   print(b)
   b.shape = (6,4)
   print(b)
def test02():
   b = np.arange(0,50,1,int)
   c = b.reshape(5,10)
   d = c.T
   print(b)
   print(c)
   print(d)
# test01()
test02()