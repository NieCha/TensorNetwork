import sys
sys.path.append("../")
import math
import numpy as np
from tensorBasis import svd_thresholding
from tools import RSE
'''
TRNNM algorithm for tensor completion ： ADMM
TENSOR-RING NUCLEAR NORM MINIMIZATION AND APPLICATION FOR VISUAL DATA COMPLETION
'''

def trnnm(tensor_observe, index, save_path=None):
    alpha=None
    rho = 0.01
    tol = 1e-5
    iter_max=1000
    shape = tensor_observe.shape
    N = len(shape)
    #### init
    if alpha is None:
        alpha = [1/N for i in range(N)]
    d = math.floor(N/2)
    X = tensor_observe.copy()
    M = [tensor_observe.copy() for i in range(N)]
    Y = [np.zeros(shape) for i in range(N)]
    permu, size, shape_= list(range(N)) * 3, np.size(tensor_observe), list(shape) * 3
    for i in range(1, iter_max+1):
        X_ = X
        for k in range(N):
            temp = np.transpose(X+ Y[k] / rho, permu[k+N+1-d : k+N+1-d+N])
            temp = np.reshape(temp, (np.prod(shape_[k+N+1-d : k+N+1]), int(size/np.prod(shape_[k+N+1-d : k+N+1]))))
            temp = svd_thresholding(temp, alpha[k] / rho)
            M[k] = np.transpose(np.reshape(temp, shape_[k+N+1-d : k+N+1-d+N]), [permu[k+N+1-d : k+N+1-d+N].index(j) for j in range(N)])
        X = (np.mean(M, axis=0) - np.mean(Y, axis=0) / rho) * (1-index) + tensor_observe
        Y = [Y[k]+rho*(X-M[k]) for k in range(N)]
        conv = RSE(X, X_)
        if (conv <= tol and i>100) or i >= iter_max:
            return X