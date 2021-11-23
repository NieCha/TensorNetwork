import sys
sys.path.append("../")
import numpy as np
from tools import RSE
from tensorBasis import svd_thresholding, unfold, fold
from decomposition.TR import TR_product

'''
TRLRF(tensor ring low rank factors) algorithm for tensor completion: ADMM
Tensor Ring Decomposition with Rank Minimization on Latent Space: An Efficient Approach for Tensor Completion
'''

def trlrf(tensor_obs, index, rank):
    iter_max = 1000
    λ = 5
    µ= 1
    µ_max = 100
    ρ = 1.01
    tol = 1e-6
    #### init
    shape = tensor_obs.shape
    N = len(shape)
    X = tensor_obs.copy()
    G_cores = [np.random.rand(rank, shape[i], rank) for i in range(N)]
    Y = [[np.zeros((rank, shape[i], rank))] * 3 for i in range(N)]
    M = [[np.zeros((rank, shape[i], rank))] * 3 for i in range(N)]
    for i in range(1, iter_max+1):
        X_ = X.copy()
        ###### update G
        for k in range(N):
            shape_core = G_cores[k].shape
            G_2 = np.reshape(np.transpose(TR_product(G_cores[k+1:] + G_cores[:k], contract_border=False), \
                            list(range(N-k, N)) + list(range(1, N-k)) + [N, 0]), [int(np.prod(shape) / shape[k]), -1])
            temp = (unfold(np.mean(M[k], axis=0) * µ + np.mean(Y[k], axis=0), 1) + λ * (unfold(X, k) @ G_2)) \
                            @ np.linalg.pinv((λ * (np.transpose(G_2) @ G_2) + 3 * µ * np.eye(G_2.shape[1], G_2.shape[1])))
            G_cores[k] = fold(temp, 1, shape_core) 
        ###### update M
            for j in range(3):
                M[k][j] = fold(svd_thresholding(unfold(G_cores[k] - Y[k][j] / µ, j), 1 / µ), j, shape_core)
        ###### update X
        X = TR_product(G_cores, contract_border=True) * (1-index) + tensor_obs
        ###### update Y
        for k in range(N):
            for j in range(3):
                Y[k][j] = Y[k][j] + µ * (M[k][j] - G_cores[k])
        µ = min(ρ * µ, µ_max)
        conv = RSE(X, X_)
        if conv <= tol or i >= iter_max:
            return X
