import sys
sys.path.append("../")
import numpy as np
from tensorNetwork.tensorNetwork import contraction
from tensorBasis import unfold, fold
from tools import RSE

'''
PAM-Based Solver for the FCTN-TC Model
Fully-Connected Tensor Network Decomposition and Its Application to Higher-Order Tensor Completion
proximal alternating minimization
Convergence of descent methods for semi-algebraic and tame problems: proximal algorithms,
forward–backward splitting, and regularized Gauss–Seidel methods
'''

def get_M_k(cores, k):
    contract_order = list(range(len(cores))) + [k]
    contract_order.pop(k)
    cores_ = [np.transpose(cores[i], contract_order) for i in contract_order]

    res = cores_[0]
    N = len(cores_)
    for i in range(1, N-1):
        #收缩节点：合并与已经收缩了的节点的edge
        shape_node = list(cores_[i].shape)
        node = np.reshape(cores_[i], [np.prod(shape_node[:i])] + shape_node[i:])
        res = np.tensordot(res, node, [[i], [0]])
        res = np.moveaxis(res, N-1, i)
        shape = res.shape
        axis, new_shape = [k for k in range(i+1)], list(shape[:i+1])
        for j in range(N-i-1):
            axis.extend([i+1+j, N+j])
            new_shape.append(shape[i+1+j] * shape[N+j])
        res = np.reshape(np.transpose(res, axis), new_shape)
    return np.reshape(res, (-1, res.shape[-1]))

def FCTN_ALS(tensor_obs, index, rank_max):
    iter_max = 1000
    tol = 1e-5
    #### init
    shape = tensor_obs.shape
    N = len(shape)
    X = tensor_obs.copy()
    cores = [np.random.random_sample([rank_max]*i + [shape[i]] + [rank_max]*(N-i-1)) for i in range(N)]
    for i in range(1, iter_max+1):
        X_ = X.copy()
        for k in range(N):
            shape_core = cores[k].shape
            M_k = get_M_k(cores, k) # I_1 I_2 I_k-1 I_k+1 I_N R_1k R_2k,R_k-1k,R_k+1k...,R_Nk
            cores[k] = fold((unfold(X, k) @ M_k) @ np.linalg.pinv(np.transpose(M_k) @ M_k), k, shape_core)
        X = contraction(cores) * (1-index) + tensor_obs
        conv = RSE(X, X_)
        if conv <= tol or i >= iter_max:
            return X

def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = np.random.random_sample(1)[0]
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value

def FCTN_PAM(tensor_ori, tensor_obs, index, rank_max, save_path=None):
    iter_max = 3000
    tol = 1e-4

    shape = list(tensor_obs.shape)
    N = len(shape)
    rho = 0.1
    R = max(1, rank_max-5)
    #### init
    cores = [np.random.random_sample([R]*i + [shape[i]] + [R]*(N-i-1)) for i in range(N)]
    X = tensor_obs.copy()
    for i in range(1, iter_max+1):
        X_ = X
        for k in range(N):
            shape_core = cores[k].shape
            M_k = get_M_k(cores, k) # I_1 I_2 I_k-1 I_k+1 I_N R_1k R_2k,R_k-1k,R_k+1k...,R_Nk 
            cores[k] = fold((unfold(X, k) @ M_k + rho * unfold(cores[k], k)) \
                        @ np.linalg.pinv(np.transpose(M_k) @ M_k + rho * np.eye(M_k.shape[1], M_k.shape[1])), k, shape_core)
        X = ((rho * X + contraction(cores)) / (1+rho)) * (1-index) + tensor_obs
        conv = RSE(X, X_)
        if conv < 1e-2 and R < rank_max:
            R += 1
            for k in range(N):
                pad = tuple([(0,1) if i!=k else (0,0) for i in range(N)])
                cores[k] = np.pad(cores[k], pad, pad_with, padder=1)
        if conv <= tol or i >= iter_max:
            return X

