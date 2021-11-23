import numpy as np

def contraction(cores, contract_order=None):
    # 收缩的顺序是通过维度置换实现
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

def generate_cores(graph):
    cores = []
    for item in range(len(graph)):
        shape = list(graph[:item, item]) + list(graph[item, item:])
        cores.append(np.random.random_sample([i if i!=0 else 1 for i in shape]))
    return cores

if __name__ == '__main__':
    graph = np.array([[3, 2, 5, 1, 6], [0, 2, 4, 2, 3], [0, 0, 4,1, 1],[0,0,0,2,2],[0,0,0,0,6]])
    cores = generate_cores(graph)
    for i in cores:
        print(i.shape)
    print(contraction(cores, contract_order=[1,4,3,0,2]).shape)

