import numpy as np
from scipy.special import expit, softmax


def softmax_grad(s): 
    jacobian_m = np.diag(s)

    for i in range(len(jacobian_m)):
        for j in range(len(jacobian_m)):
            if i == j:
                jacobian_m[i][j] = s[i] * (1-s[i])
            else: 
                jacobian_m[i][j] = -s[i]*s[j]
    return jacobian_m.dot(np.ones(len(jacobian_m)))

def ReLU(x):
    return max(0.0, x)

def ReLU_derivative(x):
    if x >= 0.0:
        return 1.0
    return 0.0

def sigmoid_derivative(x):
    return expit(x) * (1 - expit(x))

def MSE(y, yhat):
    return np.square(y - yhat).mean()

def MSE_derivative(y, yhat):
    return y-yhat

def cross_entropy(X,y):
    m = y.shape[0]
    p = softmax(X)
    log_likelihood = -np.log(p[range(m),y])
    loss = np.sum(log_likelihood) / m
    return loss

def delta_cross_entropy(X,y):
    m = y.shape[0]
    grad = softmax(X)
    grad[range(m),y] -= 1
    grad = grad/m
    return grad

