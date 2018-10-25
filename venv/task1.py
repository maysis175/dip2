import numpy as np
from mnist import MNIST
from numpy.core.multiarray import ndarray
from typing import Union

mndata = MNIST("C:\datasets")

SIZEX, SIZEY = 28, 28
PIC_LEARN = 60000
PIC_TEST = 10000
M = 100              # There are M nodes on the intermediate layer
CLASS = 10


# ========================================
# Function Definition
# ========================================

# Sigmoid function (as activate function)
def sigmoid(t):
    # Avoid stack overflow
    return np.where(t <= -710, 0, (1 / (1 + np.exp(-t))))

# Softmax function (as activate function)
def softmax(a):
    alpha = a.max()
    den_y2 = 0
    for i in range(CLASS):
        den_y2 += np.exp(a[i] - alpha)
    y2 = np.exp(a - alpha) / den_y2
    return np.argmax(y2)

def layer(seed, x, co_y, afun):
    np.random.seed(seed)
    W = np.random.normal(0, np.sqrt(1. / x.size), co_y * x.size)
    W = W.reshape(co_y, x.size)
    b = np.random.normal(0, np.sqrt(1. / x.size), co_y)
    b = b.reshape(co_y, 1)
    t = W.dot(x) + b
    return afun(t)


# =========================================
# Execution Unit
# =========================================

idx = input("Please enter a number (0-9999): ")
idx = int(idx)
if idx >= 0 and idx < PIC_TEST:
    # Preprocessing
    X, Y = mndata.load_testing()
    X = np.array(X)
    X = X.reshape((X.shape[0],SIZEX,SIZEY))
    Y = np.array(Y)

    print(Y[idx])

    #import matplotlib.pyplot as plt
    #from pylab import cm
    #plt.imshow(X[idx], cmap=cm.gray)
    #plt.show()

    # Input layer
    # Convert the image data to a vector which has (SIZEX * SIZEY) dims
    x = X[idx].ravel()
    x = np.matrix(x).T

    y1 = layer(5, x, M, sigmoid)        # Output from intermediate layer
    a = layer(10, y1, CLASS, softmax)   # Output from output layer
    print(a)

else:
    print ("Illegal Input!")
