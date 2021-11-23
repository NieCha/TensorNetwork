import numpy as np
import math

def bcirc(x):
    n1, n2, n3 = x.shape
    cir = np.zeros((n1 * n3, n2 * n3))
    s = np.eye(n3)
    for i in range(n3):
        temp = np.concatenate([s[:, i:], s[:, :i]], axis=1)
        cir += np.kron(temp, x[:, :, i])
    return cir

def bdiag(x):
    #reformulate 3 way tensor as a block diagonal matrix 
    n1, n2, n3 = x.shape
    xbdiag = np.fft.fft(np.zeros((n1 * n3, n2 * n3)))
    for i in range(n3):
        xbdiag[n1*i:n1*(i+1), n2*i:n2*(i+1)] = x[:, :, i]
    print(xbdiag.shape)
    return xbdiag

def inv_bdiag(xbdiag, n3):
    #reformulate a block diagonal matrix as 3 way tensor
    n1, n2 = xbdiag.shape / n3
    x = np.fft.fft(np.zeros((int(n1), int(n2), n3)), axis=2)
    for i in range(n3):
        x[:, :, i] = xbdiag[int(n1*i):int(n1*(i+1)), int(n2*i):int(n2*(i+1))]
    return x

def unfold(x):
    n1, n2, n3 = x.shape
    return np.concatenate([x[:,:,i] for i in range(n3)], axis = 0)

def fold(x, n1, n3):
    n1n3, n2 = x.shape
    x = x.reshape((n1n3, n2, 1))    # np.stack
    assert n1 * n3 == n1n3
    return np.concatenate([x[i*n1:(i+1)*n1, :, :] for i in range(n3)], axis = 2)

def Conjugate_transpose(x):
    n1, n2, n3 = x.shape
    res = np.zeros((n2, n1, n3))
    res[:, :, 0] = x[:, :, 0].conjugate()
    for i in range(1, n3):
        res[:, :, i] = x[:, :, n3-i].conjugate()
    return res

def t_product(A, B):
    n1, n2, n3 = A.shape        #n2=m1, n3=m3
    m1, m2, m3 = B.shape
    assert n2==m1 and n3==m3
    return fold(np.dot(bcirc(A), unfold(B)), n1, n3)

def t_product_fft(A, B):
    n1, n2, n3 = A.shape        #n2=m1, n3=m3
    m1, m2, m3 = B.shape
    assert n2==m1 and n3==m3

    A = np.fft.fft(A, axis=2)
    B = np.fft.fft(B, axis=2)
    C = np.fft.fft(np.zeros((n1, m2, n3)))     # complex form
    for i in range(math.ceil((n3+1)/2)):
        C[:, :, i] = A[:, :, i] @ B[:, :, i]
    for i in range(math.ceil((n3+1)/2), n3):
        C[:, :, i] = C[:, :, n3-i].conjugate()
    return np.fft.ifft(C, axis=2).real

def tsvd(X):
    n1, n2, n3 = X.shape
    min12 = min(n1, n2)
    X = np.fft.fft(X, axis=2)
    
    U = np.fft.fft(np.zeros((n1, n1, n3)))
    S = np.fft.fft(np.zeros((n1, n2, n3)))
    V = np.fft.fft(np.zeros((n2, n2, n3)))
    
    for i in range(math.ceil((n3+1)/2)):
        u, s, v = np.linalg.svd(X[:, :, i])
        U[:, :, i], S[:min12, :min12, i], V[:, :, i] = u, np.diag(s), v
    for i in range(math.ceil((n3+1)/2), n3):
        U[:, :, i], S[:, :, i], V[:, :, i] = U[:, :, n3-i].conjugate(), S[:, :, n3-i], V[:, :, n3-i].conjugate()
    U = np.fft.ifft(U, axis=2).real
    S = np.fft.ifft(S, axis=2).real
    V = np.fft.ifft(V, axis=2).real
    return (U, S, V)

def tensor_tubal_rank(X, tol=0):
    '''
        tensor tubal rank of a 3 way tensor
    '''
    n1, n2, n3 = X.shape
    min12 = min(n1, n2)
    X = np.fft.fft(X, axis=2).real
    S = np.zeros((min12, min12))
    for i in range(math.ceil((n3+1)/2)):
        u, s, v = np.linalg.svd(X[:, :, i])
        if i>0 and i<round(n3/2):
            S += np.diag(s)*2
        else:
            S += np.diag(s)
    S /= n3
    S = np.where(S > tol, 1, 0)
    return S.sum()

def tensor_average_rank(X):
    return np.linalg.matrix_rank(bcirc(X)) / X.shape[2]

def tubal_average_rank(X):
    n1, n2, n3 = X.shape
    X = np.fft.fft(X, axis=2).real
    r = sum([np.linalg.matrix_rank(X[:,:,i]) for i in range(n3)])
    return r / n3

def new_tensor_nuclear_norm(X):
    ## Tensor Robust Principal Component Analysis with A New T ensor Nuclear Norm
    r = tensor_tubal_rank(X)
    U, S, V = tsvd(X)
    return S[:r,:r,0].sum()

def tensor_nuclear_norm(X):
    # TNN tubal nuclear norm in 
    # Exact Tensor Completion Using t-SVD
    x = np.fft.fft(X, axis=2)
    TNN = 0
    for i in range(x.shape[2]):
        u, s, v = np.linalg.svd(x[:, :, i])
        TNN += s.sum()
    return TNN# / x.shape[2]

def tensor_spectral_norm(X):
    n1, n2, n3 = X.shape
    X = np.fft.fft(X, axis=2).real
    return max([np.linalg.svd(X[:, :, i])[1][0] for i in range(n3)])

def tensor_multi_rank(x):
    # Novel methods for multilinear data completion and de-noising based on tensor-SVD
    x = np.fft.fft(x, axis=2)
    return [np.linalg.matrix_rank(x[:,:,i]) for i in range(x.shape[2])]




