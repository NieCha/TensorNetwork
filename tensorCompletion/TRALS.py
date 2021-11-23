import sys
sys.path.append("../")
import numpy as np
from tensorBasis import unfold, fold
from tools import RSE
from decomposition.TR import TR_product

'''
Efficient Low Rank Tensor Ring Completion
'''

def TRALS(tensor_obs, index, rank):
    iter_max = 3000
    tol = 1e-5
    shape = tensor_obs.shape
    N = len(shape)
    X = tensor_obs.copy()
    lam = 5
    G_cores = [np.random.rand(rank, shape[i], rank) for i in range(N)]
    for i in range(1, iter_max+1):
        X_ = X.copy()
        ###### update G
        for k in range(N):
            shape_core = G_cores[k].shape
            G_2 = np.reshape(np.transpose(TR_product(G_cores[k+1:] + G_cores[:k], contract_border=False), \
                            list(range(N-k, N)) + list(range(1, N-k)) + [N, 0]), [int(np.prod(shape) / shape[k]), -1])
            G_cores[k] = fold(unfold(X_, k) @ G_2 @ np.linalg.pinv(np.transpose(G_2) @ G_2 + lam * np.eye(G_2.shape[1], G_2.shape[1])), 1, shape_core)
        X = TR_product(G_cores, contract_border=True) * (1-index) + tensor_obs
        conv = RSE(X, X_)
        if conv <= tol or i >= iter_max:
            return X
