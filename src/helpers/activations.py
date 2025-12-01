import numpy as np

def relu(Z):
    A = np.maximum(Z, 0)
    return A, Z

def softmax(Z):
    Z = Z - np.max(Z, axis=0, keepdims=True)
    A = np.exp(Z)
    A /= np.sum(A, axis=0, keepdims=True)
    return A, Z

def relu_backward(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0

    return dZ

def softmax_backward(dA, cache):
    return np.array(dA, copy=True)