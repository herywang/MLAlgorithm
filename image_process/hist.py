# 基于直方图均衡化的图像增强
import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = "./data/9.jpg"
# 直方图均衡化增强图像
def hist():
    image1 = cv2.imread(image_path)
    image_channel = cv2.split(image1)
    for i in range(3):
        # plt.hist(image_channel[i].ravel(), 256, [0, 256])
        # plt.show()
        cv2.equalizeHist(image_channel[i], image_channel[i])

    cv2.merge(image_channel, image1)
    return image1

# 拉普拉斯算法增强
def laplus():
    image1 = cv2.imread(image_path)
    kernel = np.array([[0,-1,0],
                       [-1,7,-1],
                       [0,-1,0]])
    dist= cv2.filter2D(image1, cv2.CV_8UC3, kernel)
    return dist

# 对数变换增强图像
def log_image():
    image1 = cv2.imread(image_path)
    image_log = np.uint8(15*np.log(np.array(image1)+1))
    cv2.normalize(image_log, image_log, 0, 255, cv2.NORM_MINMAX)
    cv2.convertScaleAbs(image_log, image_log)
    return image_log

# gamma变换图像增强
def gamma():
    image = cv2.imread(image_path)
    fgamma = 2.5
    image_gamma = np.uint8(np.power((np.array(image) / 255.0), fgamma) * 255.0)
    cv2.normalize(image_gamma, image_gamma, 0, 255, cv2.NORM_MINMAX)
    cv2.convertScaleAbs(image_gamma, image_gamma)
    return image_gamma

def lin():
    """双线性插值"""
    img = cv2.imread("./data/7.jpg", cv2.IMREAD_GRAYSCALE)  # load the gray image
    cv2.imwrite("img.jpg", img)
    h, w = img.shape[:2]

    # shrink to half of the original
    a1 = np.array([[0.5, 0, 0], [0, 0.5, 0]], np.float32)
    d1 = cv2.warpAffine(img, a1, (w, h), borderValue=125)

    # shrink to half of the original and move
    a2 = np.array([[0.5, 0, w / 4], [0, 0.5, h / 4]], np.float32)
    d2 = cv2.warpAffine(img, a2, (w, h), flags=cv2.INTER_NEAREST, borderValue=125)
    # rotate based on d2
    a3 = cv2.getRotationMatrix2D((w / 2, h / 2), 90, 1)
    d3 = cv2.warpAffine(d2, a3, (w, h), flags=cv2.INTER_LINEAR, borderValue=125)

    cv2.imshow("img", img)
    cv2.imshow("d1", d1)
    cv2.imshow("d2", d2)
    cv2.imshow("d3", d3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
if __name__ == '__main__':
    image = cv2.imread("/Users/wangheng/Desktop/Xnip2020-10-27_17-14-17.jpg")
    resize = cv2.resize(image, (108, 108))
    cv2.imwrite("/Users/wangheng/Desktop/ressize.jpg", resize)
    # shape = image.shape
    # height = shape[1] // 2
    # width = shape[0] // 2
    # image = cv2.resize(image, (height, width))
    # cv2.imwrite("data/7.jpg", image)
    # result = hist()
    # cv2.namedWindow("image", cv2.WINDOW_FREERATIO)
    # result = log_image()
    # cv2.imshow("image", result)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    lin()
    # cv2.namedWindow("image", cv2.WINDOW_FREERATIO)
    # original_image = cv2.imread(image_path, cv2.IMREAD_ANYCOLOR)
    # image = laplus()
    # image = hist()
    # image = log_image()
    # image = gamma()
    # cv2.imshow("original_image", original_image)
    # cv2.imshow("image", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()