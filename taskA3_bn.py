import numpy as np
from mnist import MNIST

# mndata = MNIST("C:\datasets")
mndata = MNIST("/export/home/016/a0161419/le4nn")

SIZEX, SIZEY = 28, 28
PIC_LEARN = 60000
PIC_TEST = 10000
M = 530                 # Intermediate layer
CLASS = 10

# Batch & Epoch
BATCH = 1024
EPOCH = 1024

ETA = 0.01          # Learning rate

EPSILON = 0.000001  # Batch Normalization


# Sigmoid function (as activate function)
def sigmoid(t):
    return 1. / (1. + np.exp(-t))

# ReLU function (as activate function)
def relu(t):
    return np.maximum(0., t)

# Softmax function (as activate function)
def softmax(a):
    alpha = a.max()
    y2 = np.exp(a - alpha) / np.sum(np.exp(a - alpha))
    return y2


# Cross entropy error
def cEntropy(y, y2):
    y = np.reshape(y, (y.shape[0], 1))
    entropy = np.sum((y * np.log(y2)))
    return entropy * -1.

# Initialize weight and biases
def setWeight(seed, x, co_y, isW):
    np.random.seed(seed)
    if isW == 1:
        W = np.random.normal(0, np.sqrt(1. / x), co_y * x)
        return W.reshape(co_y, x)
    else:
        b = np.random.normal(0, np.sqrt(1. / x), co_y)
        return b.reshape(co_y, 1)

def layer(x, W, b, actfun):
    t = W.dot(x) + b
    if actfun == 0:
        return sigmoid(t)
    elif actfun == 1:
        return relu(t)
    elif actfun == 99:
        return softmax(t)


np.set_printoptions(threshold=np.inf)

l_or_t = input("Learning or training? (Training : 0, Testing : 1) : ")
l_or_t = int(l_or_t)
if l_or_t == 0:
    # Preprocessing
    X, Y = mndata.load_training()
    X = np.array(X)
    X = X.reshape((X.shape[0], SIZEX, SIZEY))
    Y = np.array(Y)

    # Choose batches at random
    arr_idx = np.random.choice(PIC_LEARN, BATCH, replace=False)

    Xmat = X[arr_idx[0]].ravel()
    Xmat = np.reshape(Xmat, (Xmat.shape[0], 1))
    for idx in arr_idx:
        if idx != arr_idx[0]:
            Xnext = X[idx].ravel()
            Xmat = np.hstack((Xmat, np.reshape(Xnext, (Xnext.shape[0], 1))))
    Xmat = Xmat / 255.

    # Batch Normalization
    mu = np.mean(Xmat, axis=0)
    sigma = np.var(Xmat, axis=0)
    mu_mat = np.tile(mu, (SIZEX * SIZEY, 1))
    sigma_mat = np.tile(sigma, (SIZEX * SIZEY, 1))
    Xmat = (Xmat - mu_mat) / np.sqrt(sigma_mat + EPSILON)

    W1 = setWeight(1, SIZEX * SIZEY, M, 1)
    b1 = setWeight(2, SIZEX * SIZEY, M, 0)
    W2 = setWeight(3, M, CLASS, 1)
    b2 = setWeight(4, M, CLASS, 0)

    Ymat  = np.empty((CLASS, BATCH))
    Ymat1 = np.empty((M, BATCH))
    Ymat2 = np.empty((CLASS, BATCH))
    Amat1 = np.empty((M, BATCH))
    Amat2 = np.empty((CLASS, BATCH))

    for ep in range(EPOCH):
        entropy_ave = 0
        i = 0

        for idx in arr_idx:
            # Input layer
            x = X[idx].ravel()
            x = x / 255.
            x = np.reshape(x, (x.shape[0], 1))
            mu = np.mean(x, axis=0)
            sigma = np.var(x, axis=0)
            x = (x - mu) / np.sqrt(sigma + EPSILON)

            y1 = layer(x, W1, b1, 0)
            y2 = layer(y1, W2, b2, 99)

            y_arr = np.zeros(CLASS)
            y_arr[Y[idx]] = 1
            entropy_ave += cEntropy(y_arr, y2)

            Ymat[:, i:(i+1)]  = np.reshape(y_arr, (y_arr.shape[0], 1))
            Ymat1[:, i:(i+1)] = y1
            Ymat2[:, i:(i+1)] = y2

            i = i + 1

        entropy_ave = entropy_ave / BATCH
        print(str(entropy_ave))

        # Backward propagation
        En_over_a_2 = (Ymat2 - Ymat) / BATCH
        En_over_Y_1 = (W2.T).dot(En_over_a_2)
        En_over_W2  = En_over_a_2.dot(Ymat1.T)
        En_over_b2  = np.sum(En_over_a_2, axis=1)
        En_over_b2  = np.reshape(En_over_b2, (En_over_b2.shape[0], 1))

        En_over_a_1 = (1. - En_over_Y_1) * En_over_Y_1
        En_over_W1  = En_over_a_1.dot(Xmat.T)
        En_over_b1  = np.sum(En_over_a_1, axis=1)
        En_over_b1  = np.reshape(En_over_b1, (En_over_b1.shape[0], 1))

        # Optimization
        W2 = W2 - ETA * En_over_W2
        b2 = b2 - ETA * En_over_b2
        W1 = W1 - ETA * En_over_W1
        b1 = b1 - ETA * En_over_b1

    np.savez("test.npz", W1, b1, W2, b2)

elif l_or_t == 1:
    rate = 0

    # idx = input("Please select a image (0-9999) : ")
    # idx = int(idx)
    idx = 1000
    if idx >= 0 and idx < PIC_TEST:
        X, Y = mndata.load_testing()
        X = np.array(X)
        X = X.reshape((X.shape[0], SIZEX, SIZEY))
        Y = np.array(Y)

        for j in range(idx):
            x = X[j].ravel()
            x = x / 255.
            x = np.reshape(x, (x.shape[0], 1))

            loaded_para = np.load("test.npz")
            W1 = loaded_para['arr_0']
            b1 = loaded_para['arr_1']
            W2 = loaded_para['arr_2']
            b2 = loaded_para['arr_3']

            y1 = layer(x, W1, b1, 0)
            a = layer(y1, W2, b2, 99)
            print(Y[j], np.argmax(a))

            if Y[j] == np.argmax(a):
                rate = rate + 1

        print(float(rate) / idx)

else:
    print("Illegal input!")