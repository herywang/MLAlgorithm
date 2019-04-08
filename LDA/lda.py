import numpy as np
import matplotlib.pyplot as plt

np.random.seed(10)

cls1 = np.random.randn(200, 2) * np.array([1, 2]) + np.asarray([1, 5])
cls2 = np.random.randn(200, 2) * np.array([1, 1]) + np.asarray([7, 9])

figure = plt.figure()
plt.scatter(cls1[:, 0], cls1[:, 1], color='red', alpha=0.5)
plt.scatter(cls2[:, 0], cls2[:, 1], color='blue', alpha=0.5)
plt.show()
