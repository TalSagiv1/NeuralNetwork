import numpy as np
from scipy.special import expit


def softmax_grad(s): 
    jacobian_m = np.diag(s)

    for i in range(len(jacobian_m)):
        for j in range(len(jacobian_m)):
            if i == j:
                jacobian_m[i][j] = s[i] * (1-s[i])
            else: 
                jacobian_m[i][j] = -s[i]*s[j]
    return jacobian_m

def ReLU(x):
    return max(0.0, x)

def ReLU_derivative(x):
    if x >= 0.0:
        return 1.0
    return 0.0

def sigmoid_derivative(x):
    return expit(x) * (1 - expit(x))

def MSE(x):
    return np.square(x).mean()

def MSE_derivative(x):
    return x

