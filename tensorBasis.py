import numpy as np
import math
import tensorly
# reference  https://github.com/tensorly/tensorly

def tensor_to_vec(tensor):
    return np.reshape(tensor, (-1, ))

def vec_to_tensor(vec, shape):
    return np.reshape(vec, shape)

def unfold(tensor, mode):
    # numpy.moveaxis : Move axes of an array to new positions.
    res = np.moveaxis(tensor, mode, 0)
    return np.reshape(res, (tensor.shape[mode], -1))

def fold(unfolded_tensor, mode, shape):
    full_shape = list(shape)
    mode_dim = full_shape.pop(mode)
    full_shape.insert(0, mode_dim)
    return np.moveaxis(np.reshape(unfolded_tensor, full_shape), 0, mode)

def unfold_general(tensor, modes):
    modes.sort()
    res = tensor
    for i in range(len(modes)):
        res = np.moveaxis(res, modes[i], i)
    return np.reshape(res, (np.prod(modes), -1))

def fold_general(unfolded_tensor, mode, shape):
    # mode [m, n]
    full_shape = list(shape)
    full_shape.insert(0, full_shape.pop(mode[0]))
    full_shape.insert(1, full_shape.pop(mode[-1]))
    res = np.moveaxis(np.reshape(unfolded_tensor, full_shape), 1, mode[-1])
    return np.moveaxis(res, 0, mode[0])

def transpose(tensor, axes=None):
    #Reverse or permute the axes of an array; returns the modified array.
    return np.transpose(tensor, axes=axes)

def swapaxes(tensor, dim1, dim2):
    return np.swapaxes(tensor, dim1, dim2)

def tensorize(tensor, subset):
    '''
    set:[[1,3],[2,4], [0,5]]
    return I1I3 * I2I4 * I0I5 
    '''
    ori_shape = tensor.shape
    s = []
    for i in subset:
        s += i
    tensor = np.transpose(tensor, s)
    subset = [[ori_shape[j] for j in i] for i in subset]
    return np.reshape(tensor, [np.prod(i) for i in subset])

def tensor_outer_product(tensor1, tensor2):
    """Returns a generalized outer product of the two tensors
    """
    shape_1 = tensor1.shape
    shape_2 = tensor2.shape
    s1 = len(shape_1)
    s2 = len(shape_2)
    
    shape_1 = shape_1 + (1, )*s2
    shape_2 = (1, )*s1 + shape_2
    return np.reshape(tensor1, shape_1) * np.reshape(tensor2, shape_2)


def mode_dot(tensor, matrix_or_vector, mode, transpose=False):
    return tensorly.tenalg.mode_dot(tensor, matrix_or_vector, mode, transpose=transpose)

def kronecker(matrices, skip_matrix=None, reverse=False):
    return tensorly.tenalg.kronecker(matrices, skip_matrix, reverse)

def khatri_rao(matrices, weights=None, skip_matrix=None, reverse=False, mask=None):
    return tensorly.tenalg.khatri_rao(matrices, weights, skip_matrix, reverse, mask)

def soft_thresholding(tensor, threshold):
    res = np.abs(tensor) - threshold
    return np.sign(tensor) * np.where(res > 0, res, 0)

def svd_thresholding(matrix, threshold):
    U, s, V = np.linalg.svd(matrix)
    l = min(matrix.shape)
    S = np.zeros(matrix.shape)
    S[:l, :l] = np.diag(soft_thresholding(s, threshold))
    return U @ S @ V

def svd_thresholding_general(tensor, threshold):
    tensor = np.moveaxis(tensor, -1, 0)
    U, s, V = np.linalg.svd(tensor)
    l = min(tensor.shape[1:])
    S = np.zeros(tensor.shape)
    s = soft_thresholding(s, threshold)
    S[:, :l, :l] = np.stack([np.diag(s[i, :]) for i in range(tensor.shape[0])], axis=0)
    return np.moveaxis(np.einsum("ijk, ikl, ilm -> ijm" , U, S, V), 0, -1)

def tensor_inner(tensor1, tensor2, n_modes=None):
    return tensorly.tenalg.inner(tensor1, tensor2, n_modes)

def contract(tensor1, modes1, tensor2, modes2):
    return tensorly.tenalg.contract(tensor1, modes1, tensor2, modes2)
