import sys
sys.path.append("../")
import numpy as np
from tensorBasis import unfold, fold
from decomposition.TR import TR_product
from tools import RSE

'''
Tensor ring weighted optimization 
Higher-dimension tensor completion via low-rank tensor ring decomposition 2018
'''

def trwopt(tensor_obs, index, rank):
    iter_max = 2000
    tol = 1e-6
    lam = 1e-4
    #### init
    shape = tensor_obs.shape
    N = len(shape)
    X = tensor_obs.copy()
    G_cores = [np.random.rand(rank, shape[i], rank) for i in range(N)]

    for i in range(1, iter_max+1):
        X_ = X.copy()
        gradient = []
        for k in range(N):
            shape_core = G_cores[k].shape
            G_2 = np.reshape(np.transpose(TR_product(G_cores[k+1:] + G_cores[:k], contract_border=False), \
                            list(range(N-k, N)) + list(range(1, N-k)) + [N, 0]), [int(np.prod(shape) / shape[k]), -1])

            gradient.append(fold((unfold(index, k) * (unfold(G_cores[k], 1) @ np.transpose(G_2) - unfold(tensor_obs, k))) @ G_2, 1, shape_core))
        for k in range(N):
            G_cores[k] = G_cores[k] - gradient[k] * lam 
        X = TR_product(G_cores, contract_border=True) * (1-index) + tensor_obs
        conv = RSE(X, X_)
        if conv <= tol or i >= iter_max:
            return X











