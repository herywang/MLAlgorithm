# bmp图像转jpg图像实验代码(采用opencv实现, pip install opencv-python
import cv2
import numpy as np

path = "./timg.bmp"

def task1():
    image = cv2.imread(path)
    cv2.imwrite("./timg.jpg", image)

    image_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    cv2.imwrite("./timg.png", image_yuv)

def task2():
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    image_cvt = np.uint8(image * 0.25)

    cv2.imshow("image", image)
    cv2.imshow("image_cvt", image_cvt)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    #task1()
    task2()

