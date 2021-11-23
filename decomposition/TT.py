import numpy as np
import tensorly

## TT-SVD and TT_ALS
# TENSOR-TRAIN DECOMPOSITION

def TT_SVD(tensor, eplision=1e-6):
    shape = list(tensor.shape)
    d = len(shape)
    cores = []
    delta = eplision / np.sqrt(d-1) * np.linalg.norm(tensor)
    C, r = tensor, 1
    for i in range(d-1):
        u, s, v = np.linalg.svd(np.reshape(C, (int(r*shape[i]), int(C.size / (r*shape[i])))))
        r_ = min(s.shape)
        for j in range(r_):
            if s[r_-1-j:].sum() <= delta:
                r_ -= 1
            else:
                break
        u, s, v = u[:, :r_], s[:r_], v[:r_,:]
        cores.append(np.reshape(u, (r, shape[i], r_)))
        C = np.diag(s) @ v
        r = r_
    cores.append(np.reshape(C, list(C.shape)+[1]))
    return cores

def TT_product(cores, contract_border=True):
    x = np.tensordot(cores[0], cores[1], [[cores[0].ndim-1], [0]])
    for n in range(len(cores)-2):
        x = np.tensordot(x, cores[n+2], [[x.ndim - 1], [0]])
    return np.reshape(x, x.shape[1:-1]) if contract_border else x

def TT_ALS(tensor, rank, tol=1e-5, max_iter=100):
    #init
    #TT_SVD的rank是不一致的，Tensorly.tensortrain的rank是一致的（使用svd，不足填充）
    # TT_ALS 不使用SVD
    shape = tensor.shape
    d = len(shape)
    cores = [np.random.rand(1, shape[0], rank)]
    cores += [np.random.rand(rank, shape[i+1], rank) for i in range(d-2)]
    cores += [np.random.rand(rank, shape[-1], 1)]
    converged = now_iter = 0
    while not converged:
        for i in range(d):
            # subchain i+1,i+1,..i+d   0,1,2,...,i-1
            # 将后面d-1个组合
            Z_no_K = TT_product(cores[1:], contract_border=False)
            shape_k = Z_no_K.shape
            share_dim = int(shape_k[0]*shape_k[-1])
            Z_no_K = np.reshape(np.moveaxis(Z_no_K, -1, 0), (share_dim, -1))
            # Z_K_i = cores[0]
            # Z_K_i = np.reshape(np.moveaxis(Z_K_i, 2, 1), (-1, share_dim))
            tensor_mode = np.reshape(tensor, (tensor.shape[0], -1))
            mm = tensor_mode @ np.transpose(Z_no_K) @ np.linalg.pinv(Z_no_K@np.transpose(Z_no_K))
            cores[0] = np.moveaxis(np.reshape(mm, (shape[i], shape_k[-1], -1)), 1, 0)
            
            cores.append(cores.pop(0))
            tensor = np.moveaxis(tensor, 0, -1)

        RSE = np.linalg.norm(TT_product(cores) - tensor) / np.linalg.norm(tensor)
        #print(RSE)
        now_iter += 1
        if now_iter >= max_iter or RSE < tol:
            return cores

################################################3
def tensortrain(tensor, rank):
    #TT decomposition via recursive SVD  rank 是固定的
    model = tensorly.decomposition.TensorTrain(rank=5)
    return model.fit_transform(tensor)

def tt_product(tensors):
    """
    Tensor-train product
    :param tensors: tensors in the TT form
    :return: tensor
    """
    x = np.tensordot(tensors[0], tensors[1], [[tensors[0].ndim-1], [0]])
    for n in range(len(tensors)-2):
        x = np.tensordot(x, tensors[n+2], [[x.ndim - 1], [0]])
    return x


def ttd(x, chi=None):
    """
    :param x: tensor to be decomposed
    :param chi: dimension cut-off. Use QR decomposition when chi=None;
                use SVD but don't truncate when chi=-1
    :return tensors: tensors in the TT form
    :return lm: singular values in each decomposition (calculated when chi is not None)
    """
    dims = x.shape
    ndim = x.ndim
    dimL = 1
    tensors = list()
    lm = list()
    for n in range(ndim-1):
        if chi is None:  # No truncation
            q, x = np.linalg.qr(x.reshape(dimL*dims[n], -1))
            dimL1 = x.shape[0]
        else:
            q, s, v = np.linalg.svd(x.reshape(dimL*dims[n], -1))
            if chi > 0:
                dc = min(chi, s.size)
            else:
                dc = s.size
            q = q[:, :dc]
            s = s[:dc]
            lm.append(s)
            x = np.diag(s).dot(v[:dc, :])
            dimL1 = dc
        tensors.append(q.reshape(dimL, dims[n], dimL1))
        dimL = dimL1
    tensors.append(x.reshape(dimL, dims[-1]))
    tensors[0] = tensors[0][0, :, :]
    return tensors, lm


def entanglement_entropy(lm, tol=1e-20):
    lm /= np.linalg.norm(lm)
    lm = lm[lm > tol]
    ent = -2 * (lm ** 2).T.dot(np.log(lm))
    return ent

if __name__ == '__main__':
    a = np.random.rand(6,3,4,5)
    cores = TT_ALS(a, 18)
    for i in cores:
        print(i.shape)
    b = TT_product(cores)
    print(b.shape)
    print(np.linalg.norm(a-b)/np.linalg.norm(a))










