import numpy as np
import math

# TR_SVD
def trunk_svd(matrix, delta):
    u, s, v = np.linalg.svd(matrix)
    r = min(s.shape)
    for j in range(r):
        if s[r-1-j:].sum() <= delta:
            r -= 1
        else:
            return u[:, :r], s[:r], v[:r, :], r

def get_r(r1r2):
    factors = [i for i in range(1, r1r2+1) if r1r2%i==0]
    l = len(factors) // 2
    return factors[l-1], factors[l]

def TR_SVD(tensor, eplision=1e-6):
    shape = list(tensor.shape)
    d = len(shape)
    cores = []
    delta = eplision / np.sqrt(d) * np.linalg.norm(tensor)
    delta = [delta * math.sqrt(2) if i==0 else delta for i in range(d)]
    C = tensor
    u, s, v, r1r2 = trunk_svd(np.reshape(C, (shape[0], -1)), delta[0])
    r1, r = get_r(r1r2)
    cores.append(np.transpose(np.reshape(u, (shape[0], r1, r)), [1, 0, 2]))
    C = np.transpose(np.reshape(np.diag(s)@v, (r1, r, -1)), (1, 2, 0))
    for i in range(1, d-1):
        u, s, v, r_ = trunk_svd(np.reshape(C, (int(r*shape[i]), -1)), delta[i])
        r_ = min(s.shape)

        cores.append(np.reshape(u, (r, shape[i], r_)))
        C = np.diag(s) @ v
        r = r_
    cores.append(np.reshape(C, (r_, shape[-1], r1)))
    return cores

def TR_product(cores, contract_border=True):
    if len(cores)==2:
        if contract_border:
            return np.tensordot(cores[0], cores[1], [[0, 2], [2, 0]])
        else:
            return np.tensordot(cores[0], cores[1], [[cores[0].ndim-1], [0]])

    x = np.tensordot(cores[0], cores[1], [[cores[0].ndim-1], [0]])
    for n in range(len(cores)-3):
        x = np.tensordot(x, cores[n+2], [[x.ndim - 1], [0]])
    if contract_border:
        return np.tensordot(x, cores[-1], [[0, x.ndim - 1], [2, 0]])
    else:
        return np.tensordot(x, cores[-1], [[x.ndim - 1], [0]])

def TR_ALS(tensor, rank, tol=1e-5, max_iter=100):
    shape = tensor.shape
    d = len(shape)
    cores = [np.random.rand(rank, shape[i], rank) for i in range(d)]
    converged = now_iter = 0
    while not converged:
        for i in range(d):
            # subchain i+1,i+1,..i+d   0,1,2,...,i-1
            Z_no_K = TR_product(cores[1:], contract_border=False)
            
            shape_k = Z_no_K.shape
            share_dim = int(shape_k[0]*shape_k[-1])
            Z_no_K = np.reshape(np.moveaxis(Z_no_K, -1, 0), (share_dim, -1))

            tensor_mode = np.reshape(tensor, (tensor.shape[0], -1))
            mm = tensor_mode @ np.transpose(Z_no_K) @ np.linalg.pinv(Z_no_K@np.transpose(Z_no_K))
            cores[0] = np.moveaxis(np.reshape(mm, (shape[i], shape_k[-1], -1)), 1, 0)
            
            cores.append(cores.pop(0))
            tensor = np.moveaxis(tensor, 0, -1)

        RSE = np.linalg.norm(TR_product(cores) - tensor) / np.linalg.norm(tensor)
        print(RSE)
        now_iter += 1
        if now_iter >= max_iter or RSE < tol:
            return cores