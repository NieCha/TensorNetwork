import sys
sys.path.append("../")
import numpy as np
from tensorBasis import unfold, fold
from tools import RSE
from decomposition.TR import TR_product

'''
Efficient Low Rank Tensor Ring Completion
'''

def TTALS(tensor_obs, index, rank):
    iter_max = 1000
    tol = 1e-5
    #### init
    shape = tensor_obs.shape
    N = len(shape)
    X = tensor_obs.copy()
    if type(rank) == int:
        rank = [rank] * N
    cores = [np.random.rand(1, shape[0], rank[0])]
    cores += [np.random.rand(rank[i], shape[i+1], rank[i+1]) for i in range(N-2)]
    cores += [np.random.rand(rank[-1], shape[-1], 1)]
    for i in range(1, iter_max+1):
        X_ = X.copy()
        ###### update G
        for k in range(N):
            shape_core = cores[k].shape
            G_2 = np.reshape(np.transpose(TR_product(cores[k+1:] + cores[:k], contract_border=False), \
                            list(range(N-k, N)) + list(range(1, N-k)) + [N, 0]), [int(np.prod(shape) / shape[k]), -1])
            cores[k] = fold(unfold(X_, k) @ G_2 @ np.linalg.pinv(np.transpose(G_2) @ G_2), 1, shape_core)

        X = TR_product(cores, contract_border=True) * (1-index) + tensor_obs
        conv = RSE(X, X_)
        if conv <= tol or i >= iter_max:
            return X

