import numpy as np
from mnist import MNIST

# mndata = MNIST("C:\datasets")
mndata = MNIST("/export/home/016/a0161419/le4nn")

SIZEX, SIZEY = 28, 28
PIC_LEARN = 60000
PIC_TEST = 10000
M = 500                 # Intermediate layer
CLASS = 10

# Dropout
# 0: off, 1: on
DROPOUT = 0

# Activate function
# 0: sigmoid, 1: ReLU
ACTFUN = 1

# Optimization techniques
# 0: SGD, 1: Momentum SGD, 2: AdaGrad, 3: RMSProp, 4: AdaDelta, 5: Adam
OPTTECH = 5

ALPHA = 0.9         # for Momentum SGD
ETA_ADAG = 0.001    # for AdaGrad
ETA = 0.01          # Learning rate

RHO = 0.9           # RMSProp
RHO_DELTA = 0.95    # AdaDelta
EPSILON = 0.00000001    # RMSProp
EPSILON_D = 0.000001    # AdaDelta
ALPHA_ADAM = 0.001  # Adam
BETA_1 = 0.9
BETA_2 = 0.999
EPSILON_A = 0.00000001  # Adam


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

# AdaGrad
def adagrad(h, W, Wgrad):
    h = h + Wgrad * Wgrad
    W = W - ETA_ADAG * (1. / np.sqrt(h)) * Wgrad
    return h, W

# RMSProp
def rmsprop(h, W, Wgrad):
    h = RHO * h + (1 - RHO) * Wgrad * Wgrad
    W = W - ETA_ADAG / (np.sqrt(h) + EPSILON) * Wgrad
    return h, W

# AdaDelta
def adadelta(h, s, W, Wgrad):
    h = RHO_DELTA * h + (1 - RHO_DELTA) * Wgrad * Wgrad
    deltaW = -1. * np.sqrt((s + EPSILON_D) / (h + EPSILON_D)) * Wgrad
    s = RHO_DELTA * s + (1 - RHO_DELTA) * deltaW * deltaW
    W = W + deltaW
    return h, s, W

# Adam
def adam(t, m, v, W, Wgrad):
    t = t + 1
    m = BETA_1 * m + (1 - BETA_1) * Wgrad
    v = BETA_2 * v + (1 - BETA_2) * Wgrad * Wgrad
    mhat = m / (1 - BETA_1 ** t)
    vhat = v / (1 - BETA_2 ** t)
    W = W - ALPHA_ADAM * mhat / (np.sqrt(vhat) + EPSILON_A)
    return t, m, v, W


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


BATCH = 512
EPOCH = 784
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

    W1 = setWeight(1, SIZEX * SIZEY, M, 1)
    b1 = setWeight(2, SIZEX * SIZEY, M, 0)
    W2 = setWeight(3, M, CLASS, 1)
    b2 = setWeight(4, M, CLASS, 0)

    Ymat  = np.empty((CLASS, BATCH))
    Ymat1 = np.empty((M, BATCH))
    Ymat2 = np.empty((CLASS, BATCH))
    Amat1 = np.empty((M, BATCH))
    Amat2 = np.empty((CLASS, BATCH))

    if OPTTECH == 1:
        deltaW1 = np.zeros((M, SIZEX * SIZEY))
        deltaW2 = np.zeros((CLASS, M))
        deltab1 = np.zeros((M, 1))
        deltab2 = np.zeros((CLASS, 1))
    if OPTTECH == 2:
        h_W1 = np.full((M, SIZEX * SIZEY), 0.00000001)
        h_W2 = np.full((CLASS, M), 0.00000001)
        h_b1 = np.full((M, 1), 0.00000001)
        h_b2 = np.full((CLASS, 1), 0.00000001)
    if OPTTECH == 3:
        h_W1 = np.zeros((M, SIZEX * SIZEY))
        h_W2 = np.zeros((CLASS, M))
        h_b1 = np.zeros((M, 1))
        h_b2 = np.zeros((CLASS, 1))
    if OPTTECH == 4:
        h_W1 = np.zeros((M, SIZEX * SIZEY))
        h_W2 = np.zeros((CLASS, M))
        h_b1 = np.zeros((M, 1))
        h_b2 = np.zeros((CLASS, 1))
        s_W1 = np.zeros((M, SIZEX * SIZEY))
        s_W2 = np.zeros((CLASS, M))
        s_b1 = np.zeros((M, 1))
        s_b2 = np.zeros((CLASS, 1))
    if OPTTECH == 5:
        t_W1, t_W2, t_b1, t_b2 = 0, 0, 0, 0
        m_W1 = np.zeros((M, SIZEX * SIZEY))
        m_W2 = np.zeros((CLASS, M))
        m_b1 = np.zeros((M, 1))
        m_b2 = np.zeros((CLASS, 1))
        v_W1 = np.zeros((M, SIZEX * SIZEY))
        v_W2 = np.zeros((CLASS, M))
        v_b1 = np.zeros((M, 1))
        v_b2 = np.zeros((CLASS, 1))

    for ep in range(EPOCH):
        entropy_ave = 0
        i = 0

        for idx in arr_idx:
            # Input layer
            x = X[idx].ravel()
            x = x / 255.
            x = np.reshape(x, (x.shape[0], 1))

            y1 = layer(x, W1, b1, ACTFUN)
            y2 = layer(y1, W2, b2, 99)
            if ACTFUN == 1:
                a1 = W1.dot(x) + b1
                a2 = W2.dot(y1) + b2
                Amat1[:, i:(i+1)] = a1
                Amat2[:, i:(i+1)] = a2

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

        if ACTFUN == 0:
            En_over_a_1 = (1. - En_over_Y_1) * En_over_Y_1
        elif ACTFUN == 1:
            En_over_a_1 = np.where((Amat1 > 0), En_over_Y_1, float(0))

        En_over_W1  = En_over_a_1.dot(Xmat.T)
        En_over_b1  = np.sum(En_over_a_1, axis=1)
        En_over_b1  = np.reshape(En_over_b1, (En_over_b1.shape[0], 1))

        if OPTTECH == 0:
            W2 = W2 - ETA * En_over_W2
            b2 = b2 - ETA * En_over_b2
            W1 = W1 - ETA * En_over_W1
            b1 = b1 - ETA * En_over_b1
        elif OPTTECH == 1:
            deltaW2 = ALPHA * deltaW2 - ETA * En_over_W2
            W2 = W2 + deltaW2
            deltab2 = ALPHA * deltab2 - ETA * En_over_b2
            b2 = b2 + deltab2
            deltaW1 = ALPHA * deltaW1 - ETA * En_over_W1
            W1 = W1 + deltaW1
            deltab1 = ALPHA * deltab1 - ETA * En_over_b1
            b1 = b1 + deltab1
        elif OPTTECH == 2:
            h_W2, W2 = adagrad(h_W2, W2, En_over_W2)
            h_b2, b2 = adagrad(h_b2, b2, En_over_b2)
            h_W1, W1 = adagrad(h_W1, W1, En_over_W1)
            h_b1, b1 = adagrad(h_b1, b1, En_over_b1)
        elif OPTTECH == 3:
            h_W2, W2 = rmsprop(h_W2, W2, En_over_W2)
            h_b2, b2 = rmsprop(h_b2, b2, En_over_b2)
            h_W1, W1 = rmsprop(h_W1, W1, En_over_W1)
            h_b1, b1 = rmsprop(h_b1, b1, En_over_b1)
        elif OPTTECH == 4:
            h_W2, s_W2, W2 = adadelta(h_W2, s_W2, W2, En_over_W2)
            h_b2, s_b2, b2 = adadelta(h_b2, s_b2, b2, En_over_b2)
            h_W1, s_W1, W1 = adadelta(h_W1, s_W1, W1, En_over_W1)
            h_b1, s_b1, b1 = adadelta(h_b1, s_b1, b1, En_over_b1)
        elif OPTTECH == 5:
            t_W2, m_W2, v_W2, W2 = adam(t_W2, m_W2, v_W2, W2, En_over_W2)
            t_b2, m_b2, v_b2, b2 = adam(t_b2, m_b2, v_b2, b2, En_over_b2)
            t_W1, m_W1, v_W1, W1 = adam(t_W1, m_W1, v_W1, W1, En_over_W1)
            t_b1, m_b1, v_b1, b1 = adam(t_b1, m_b1, v_b1, b1, En_over_b1)


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

            y1 = layer(x, W1, b1, ACTFUN)
            a = layer(y1, W2, b2, 99)
            print(Y[j], np.argmax(a))

            if Y[j] == np.argmax(a):
                rate = rate + 1

        print(float(rate) / idx)

else:
    print("Illegal input!")
