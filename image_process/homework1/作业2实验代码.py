import cv2
import numpy as np

s = 20          #划分多少个颜色级别
c = 255 // s    #每个颜色级别的颜色为多少
width = 30

def rgb_channel():
    # 三通道彩色图像测试
    # 每个颜色级别宽度为30个像素,因此用40*s
    image = np.zeros((190, width * s, 3))
    for i in range(s):
        image[0:50, i * width:(i + 1) * width, 0] = i*c
        image[70:120, i * width:(i + 1) * width, 1] = i*c
        image[140:190, i * width:(i + 1) * width, 2] = i*c
    return np.uint8(image)

def gray_channel():
    image = np.zeros((50, width*s))
    for i in range(s):
        image[:, i*width:(i+1)*width] = i*c
    return np.uint8(image)

#image = rgb_channel()   #三通道彩色图像测试
image = gray_channel()

cv2.namedWindow("image", cv2.WINDOW_FREERATIO)
cv2.imshow("image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
