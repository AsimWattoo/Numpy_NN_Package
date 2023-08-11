import numpy as np

def softmax(z):
    return np.exp(z) / np.reshape(np.sum(np.exp(z), 1), (-1, 1))

def softmax_comps(z, comps):
    return np.exp(z) / np.sum(np.exp(comps))

def softmax_prime(z):
    # z = softmax(z)
    # Number of records
    m = z.shape[0]
    n = z.shape[1]
    prime = np.zeros((m, n))
    for i in range(m):
        comps = z[i, :]
        for j in range(n):
            if i == j:
                prime[i, j] = z[i, j], (1 - z[i, j]) 
            else:
                prime[i, j] = -z[i, j] * z[i, j]
    return prime

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_prime(z):
    return np.multiply(sigmoid(z), (1 - sigmoid(z)))

def dummy_activation(z):
    return z