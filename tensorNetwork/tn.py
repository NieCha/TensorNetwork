'''
numpy 随机参数 类型
使用tensornetwork完成对张量图的收缩
创建MPS
'''

import tensornetwork as tn
import numpy as np


def block(dimensions):
    '''Construct a new matrix for the MPS with random numbers from 0 to 1'''
    size = tuple([x for x in dimensions])
    return np.random.random_sample(size)

def create_MPS(rank, dimension, bond_dim, boundry='open'):
    '''Build the MPS tensor
    rnak : 个数
    dimension 收缩后的维度
    bond_dim
    open : TT
    close Tensor ring
    '''
    if boundry == 'open':
        mps = [tn.Node( block([dimension[0], bond_dim]) )] + \
            [tn.Node( block([bond_dim, dimension[i+1], bond_dim])) for i in range(rank-2)] + \
            [tn.Node( block([bond_dim, dimension[-1]]) )]
        #connect edges to build mps
        connected_edges=[]
        conn=mps[0][1]^mps[1][0]
        connected_edges.append(conn)
        for k in range(1,rank-1):
            conn=mps[k][2]^mps[k+1][0]
            connected_edges.append(conn)
    else:
        mps = [tn.Node(block([bond_dim, dimension[0], bond_dim]))] + \
            [tn.Node(block([bond_dim, dimension[i+1], bond_dim])) for i in range(rank-2)] + \
            [tn.Node(block([bond_dim, dimension[-1], bond_dim]))]
        #connect edges to build mps
        connected_edges=[]
        for k in range(rank-1):
            conn=mps[k][2]^mps[k+1][0]
            connected_edges.append(conn)
        conn=mps[-1][2]^mps[0][0]
        connected_edges.append(conn)
    return mps, connected_edges

def create_FCN(graph):
    '''
    para : graph 上三角矩阵，两个node没有连接使用0替代
    '''
    fcn = []
    for item in range(len(graph)):
        shape = list(graph[:item, item]) + list(graph[item, item:])
        factor = tn.Node(block([i if i!=0 else 1 for i in shape]))
        fcn.append(factor)
    connected_edges=[]
    for i in range(len(graph)-1):
        for j in range(i+1, len(graph)):
            conn=fcn[i][j]^fcn[j][i]
            connected_edges.append(conn)
    return fcn, connected_edges

def contract_FCN(cores, graph):
    fcn = [tn.Node(i) for i in cores]
    for i in range(len(graph)-1):
        for j in range(i+1, len(graph)):
            fcn[i][j]^fcn[j][i]
    return tn.contractors.auto(fcn, [fcn[i][i] for i in range(len(fcn))])

if __name__ == '__main__':
    # mps, edge = create_MPS(3, [2,2,4], 2, boundry='close')
    # result = tn.contractors.auto(mps, [i[1] for i in mps]) #node  output_edge_order 收缩节点的顺序
    # #result = tn.contractors.optimal(mps, ignore_edge_order=True)
    # print(result.tensor.shape)
    fcn, edge = create_FCN(np.array([[3, 2, 3], [0, 2, 4], [0, 0, 4]]))
    print(contract_FCN(fcn, np.array([[3, 2, 3], [0, 2, 4], [0, 0, 4]])).shape)
    result = tn.contractors.auto(fcn, [fcn[i][i] for i in range(len(fcn))]) #node  output_edge_order 收缩节点的顺序
    #result = tn.contractors.optimal(mps, ignore_edge_order=True)
    print(result.tensor.shape)