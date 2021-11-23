import numpy as np
from functools import reduce

def calPara(shape):
    return reduce(np.multiply, shape, 1)

def calParaGraph(graph):
    para = 0
    for i in range(len(graph)):
        core = np.concatenate([graph[0:i, i], graph[i, i:]], axis=0)
        para += calPara([i for i in core if i != 0])
    return para

def gene_rank(tensor, i, j, shape, kappa):
    i, j = min([i,j]), max([i,j])
    ten = np.swapaxes(tensor, i, 0)
    ten = np.swapaxes(ten, j, 1).reshape(shape[i], shape[j], -1)
    return nuclearNorm(ten, kappa)

def nuclearNorm(X, kappa, tol=0.0):
    n1, n2, n3 = X.shape
    min12 = min(n1, n2)
    S = np.zeros((min12, min12))
    for i in range(n3):
        u, s, v = np.linalg.svd(X[:, :, i])
        S += np.diag(s)
    # S = S ** 2
    S /= S.sum()
    for i in range(min12):
        if S[:i+1, :i+1].sum() >= kappa:
            return i+1
    return i+1

def tensorStructure(tensor, kappa=None, CR=None):
    shape = tensor.shape
    graph = np.diag(shape)
    node = len(shape)
    if not CR:
        for i in range(node-1):
            for j in range(i+1, node):
                rank = gene_rank(tensor, i, j, shape, kappa)
                graph[i][j] = graph[j][i] = rank
        return graph, kappa
    else:
        k_min, k_max = 0, 1
        v = {}
        for i in range(node-1):
            for j in range(i+1, node):
                v[str(i)+str(j)] = get_vector(tensor, i, j, shape)
        k_min = min([v[k][0] for k in v.keys()])
        graph_min = graph.copy()
        graph_max = graph.copy() + 1
        while True:
            graph_now = graph.copy()
            kappa = (k_max + k_min)/2
            for i in range(node):
                for j in range(i+1, node):
                    v_ij = v[str(i)+str(j)]
                    for l in range(len(v_ij)):
                        if (v_ij[:l+1]).sum() >= kappa:
                            graph_now[i, j] = (l+1) if (l+1) != 1 else 0
                            break
            if calPara(shape) / calParaGraph(graph_now) > CR:
                k_min = kappa
                graph_min = graph_now
            else:
                k_max = kappa
                graph_max = graph_now
            if graph_max.sum() - graph_min.sum() <=1 or (k_max - k_min) < 1e-6:
                return graph_min, k_min
                
def get_vector(tensor, i, j, shape):
    ten = np.swapaxes(tensor, i, 0)
    ten = np.swapaxes(ten, j, 1).reshape(shape[i], shape[j], -1)
    n1, n2, n3 = ten.shape
    min12 = min(n1, n2)
    S = np.zeros((min12, min12))
    for i in range(n3):
        u, s, v = np.linalg.svd(ten[:, :, i])
        S += np.diag(s)
    S = S ** 2
    return np.diag(S) / np.sum(S)

def RSE(x, x_pre, tol=0):
    return np.linalg.norm(x - x_pre) / (np.linalg.norm(x_pre)+tol)

def unfold(tensor, mode):
    res = np.moveaxis(tensor, mode, 0)
    return np.reshape(res, (tensor.shape[mode], -1))

def fold(unfolded_tensor, mode, shape):
    full_shape = list(shape)
    mode_dim = full_shape.pop(mode)
    full_shape.insert(0, mode_dim)
    return np.moveaxis(np.reshape(unfolded_tensor, full_shape), 0, mode)

def contract_graph(cores, graph=None, contract_order=None):
    if contract_order:
        cores = [np.transpose(cores[i], contract_order) for i in contract_order]
    res = cores[0]
    N = len(cores)
    for i in range(1, N):
        shape_node = list(cores[i].shape)
        node = np.reshape(cores[i], [np.prod(shape_node[:i])] + shape_node[i:])
        res = np.tensordot(res, node, [[i], [0]])
        res = np.moveaxis(res, N-1, i)
        shape = res.shape
        axis, new_shape = [k for k in range(i+1)], list(shape[:i+1])
        for j in range(N-i-1):
            axis.extend([i+1+j, N+j])
            new_shape.append(shape[i+1+j] * shape[N+j])
        res = np.reshape(np.transpose(res, axis), new_shape)
    return np.transpose(res, [contract_order.index(i) for i in range(N)]) if contract_order else res

def Comosition_except_k(cores, k):
    contract_order = list(range(len(cores))) + [k]
    contract_order.pop(k)
    cores_ = [np.transpose(cores[i], contract_order) for i in contract_order]

    res = cores_[0]
    N = len(cores_)
    for i in range(1, N-1):
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

def FCTN_ALS(weight, graph, iter=300):
    iter_max = iter
    tol = 1e-5
    #### init
    shape = weight.shape
    N = len(shape)
    cores = [np.random.randn(*np.concatenate([graph[0:i, i], graph[i, i:]], axis=0)) for i in range(N)]
    for i in range(1, iter_max+1):
        for k in range(N):
            shape_core = cores[k].shape
            M_k = Comosition_except_k(cores, k)
            cores[k] = fold((unfold(weight, k) @ M_k) @ np.linalg.pinv(np.transpose(M_k) @ M_k), k, shape_core)
        X = contract_graph(cores)
        rse = RSE(X, weight)
        
        if i % 100 == 0:
            print('iter : ', i,
                  ', report rse : ', rse)
        if rse <= tol or i >= iter_max:
            return cores

def adapDecomp(tensor, kappa=None, CR=None):
    graph, kappa = tensorStructure(tensor, kappa=kappa, CR=CR)
    cores = FCTN_ALS(tensor, graph, iter=300)
    return cores
