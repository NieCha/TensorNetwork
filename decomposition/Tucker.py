import numpy as np
import tensorly
import copy
import torch as tc

def hosvd(x):
    """
    :param x: 待分解的张量
    :return G: 核张量
    :return U: 变换矩阵
    :return lm: 各个键约化矩阵的本征谱
    """
    ndim = x.ndim
    U = list()  # 用于存取各个变换矩阵
    lm = list()  # 用于存取各个键约化矩阵的本征谱
    x = tc.from_numpy(x)
    for n in range(ndim):
        index = list(range(ndim))
        index.pop(n)
        _mat = tc.tensordot(x, x, [index, index])
        _lm, _U = tc.symeig(_mat, eigenvectors=True)
        lm.append(_lm.numpy())
        U.append(_U)
    # 计算核张量
    G = tucker_product(x, U)
    U1 = [u.numpy() for u in U]
    return G, U1, lm

def tucker_product(x, U, dim=1):
    """
    :param x: 张量
    :param U: 变换矩阵
    :param dim: 收缩各个矩阵的第几个指标
    :return G: 返回Tucker乘积的结果
    """
    ndim = x.ndim
    if type(x) is not tc.Tensor:
        x = tc.from_numpy(x)

    U1 = list()
    for n in range(len(U)):
        if type(U[n]) is not tc.Tensor:
            U1.append(tc.from_numpy(U[n]))
        else:
            U1.append(U[n])

    ind_x = ''
    for n in range(ndim):
        ind_x += chr(97 + n)
    ind_x1 = ''
    for n in range(ndim):
        ind_x1 += chr(97 + ndim + n)
    contract_eq = copy.deepcopy(ind_x)
    for n in range(ndim):
        if dim == 0:
            contract_eq += ',' + ind_x[n] + ind_x1[n]
        else:
            contract_eq += ',' + ind_x1[n] + ind_x[n]
    contract_eq += '->' + ind_x1
    # print(x.shape, U[0].shape, U[1].shape, U[2].shape)
    # print(type(contract_eq), contract_eq)
    G = tc.einsum(contract_eq, [x] + U1)
    G = G.numpy()
    return G

# HOOI  higher-order orthogonal iteration
#使用HOSVD初始化，然后采用ALS迭代求解

def HOOI(tensor, rank=None, n_iter_max=100, init='svd', svd='numpy_svd', tol=0.0001):
    return tensorly.decomposition.Tucker(tensor, rank=None, n_iter_max=100, init='svd', svd='numpy_svd', tol=0.0001)
