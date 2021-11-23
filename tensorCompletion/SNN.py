import sys
sys.path.append("../")
import numpy as np
from tensorBasis import svd_thresholding, unfold, fold
from tools import RSE
'''
Tensor Completion for Estimating Missing Values in Visual Datas
'''

def SiLRTC(tensor_observe, index):
    iter_max = 300
    tol = 1e-6
    shape = tensor_observe.shape
    N = len(shape)
    alpha = [1/N for i in range(N)]
    gamma = 10
    beta = [1/gamma for i in range(N)]
    X = tensor_observe.copy()
    M = [np.random.random_sample(shape) for i in range(N)]
    for i in range(1, iter_max+1):
        X_ = X.copy()
        for k in range(N):
            M[k] = fold(svd_thresholding(unfold(X, k), alpha[k] / beta[k]), k, shape)
        X = (np.sum([beta[k] * M[k] for k in range(N)], axis=0) / np.sum(beta, axis=0)) * (1-index) + tensor_observe
        conv = RSE(X, X_)
        if conv <= tol and i >= iter_max:
            return X

def FaLRTC(T, index):
    pass

def HaLRTC(tensor_ori, tensor_observe, index, save_path=None):
    iter_max = 300
    tol = 1e-6
    shape = tensor_observe.shape
    N = len(shape)
    alpha = [1/N for i in range(N)]
    rho = 1e-2  # 1e-3...
    X = tensor_observe.copy()
    M = [np.random.random_sample(shape) for i in range(N)]
    Y = [np.zeros(shape) for i in range(N)]
    for i in range(1, iter_max+1):
        X_ = X.copy()
        for k in range(N):
            print(unfold(X + Y[k] / rho, k).shape)
            M[k] = fold(svd_thresholding(unfold(X + Y[k] / rho, k), alpha[k] / rho), k, shape)
        X = (np.sum([M[k] - Y[k] / rho for k in range(N)], axis=0) / N) * (1-index) + tensor_observe
        for k in range(N):
            Y[k] = Y[k] + rho * (X - M[k])
        rho *= 1.1
        conv = RSE(X, X_)
        if conv <= tol and i >= iter_max:
            return X