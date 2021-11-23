import numpy as np
import tensorly
import copy

def CP_ALS(tensor, rank):
    #http://tensorly.org/stable/modules/generated/tensorly.decomposition.CP.html#tensorly.decomposition.CP
    return tensorly.decomposition.CP(tensor, rank)

def eig0(mat, it_time=100, tol=1e-15):
    """
    :param mat: 输入矩阵（实对称阵）
    :param it_time: 最大迭代步数
    :param tol: 收敛阈值
    :return lm: （绝对值）最大本征值
    :return v1: 最大本征向量
    """
    # 初始化向量
    v1 = np.random.randn(mat.shape[0],)
    v0 = copy.deepcopy(v1)
    lm = 1
    for n in range(it_time):  # 开始循环迭代
        v1 = mat.dot(v0)  # 计算v1 = M V0
        lm = np.linalg.norm(v1)  # 求本征值
        v1 /= lm  # 归一化v1
        # 判断收敛
        conv = np.linalg.norm(v1 - v0)
        if conv < tol:
            break
        else:
            v0 = copy.deepcopy(v1)
    return lm, v1

def svd0(mat, it_time=100, tol=1e-15):
    """
    Recursive algorithm to find the dominant singular value and vectors
    :param mat: input matrix (assume to be real)
    :param it_time: max iteration time
    :param tol: tolerance of error
    :return u: the dominant left singular vector
    :return s: the dominant singular value
    :return v: the dominant right singular vector
    """
    dim0, dim1 = mat.shape
    # 随机初始化奇异向量
    u, v = np.random.randn(dim0, ), np.random.randn(dim1, )
    # 归一化初始向量
    u, v = u/np.linalg.norm(u), v/np.linalg.norm(v)
    s = 1

    for t in range(it_time):
        # 更新v和s
        v1 = u.dot(mat)
        s1 = np.linalg.norm(v1)
        v1 /= s1
        # 更新u和s
        u1 = mat.dot(v1)
        s1 = np.linalg.norm(u1)
        u1 /= s1
        # 计算收敛程度
        conv = np.linalg.norm(u - u1) / dim0 + np.linalg.norm(v - v1) / dim1
        u, s, v = u1, s1, v1
        # 判断是否跳出循环
        if conv < tol:
            break
    return u, s, v

# rank-1 decomposition
def rank1decomp(x, it_time=100, tol=1e-15):
    """
    :param x: 待分解的张量
    :param it_time: 最大迭代步数
    :param tol: 迭代终止的阈值
    :return vs: 储存rank-1分解各个向量的list
    :return k: rank-1系数
    """
    ndim = x.ndim  # 读取张量x的阶数
    dims = x.shape  # 读取张量x各个指标的维数

    # 初始化vs中的各个向量并归一化
    vs = list()  # vs用以储存rank-1分解得到的各个向量
    for n in range(ndim):
        _v = np.random.randn(dims[n])
        vs.append(_v / np.linalg.norm(_v))
    k = 1

    for t in range(it_time):
        vs0 = copy.deepcopy(vs)  # 暂存各个向量以计算收敛情况
        for _ in range(ndim):
            # 收缩前(ndim-1)个向量，更新最后一个向量
            x1 = copy.deepcopy(x)
            for n in range(ndim-1):
                x1 = np.tensordot(x1, vs[n], [[0], [0]])
            # 归一化得到的向量，并更新常数k
            k = np.linalg.norm(x1)
            x1 /= k
            # 将最后一个向量放置到第0位置
            vs.pop()
            vs.insert(0, x1)
            # 将张量最后一个指标放置到第0位置
            x = x.transpose([ndim-1] + list(range(ndim-1)))
        # 计算收敛情况
        conv = np.linalg.norm(np.hstack(vs0) - np.hstack(vs))
        if conv < tol:
            break
    return vs, k