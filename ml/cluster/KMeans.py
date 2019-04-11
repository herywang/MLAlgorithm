import cv2
import numpy as np

class KMeans(object):
    def __init__(self):
        pass

    @staticmethod
    def cluster(k, data):
        clusters = np.zeros([k, data.shape[1]])
        result = np.zeros([data.shape[0]])
        for i in range(k):
            clusters[i] = data[i]
        for i in range(1000):
            for i in range(data.shape[0]):
                result[i] = KMeans.distance(data[i], clusters)


    @staticmethod
    def distance(data, cluster):
        res = []
        for point in cluster:
            res.append(np.sqrt(np.dot((data - point), (data - point))))
        return np.argmin(res)

image = cv2.imread("./k-means.png", cv2.IMREAD_ANYCOLOR)
rows, cols = image.shape[0], image.shape[1]
data = image.reshape(-1, 3)
print(data.shape, type(data))
KMeans.cluster(4, data)


