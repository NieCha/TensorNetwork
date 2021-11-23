import numpy as np
import torch
import torch.nn as nn

class TNGraph(nn.Module):
    def __init__(self, graph, contract_order=None):
        super(TNGraph, self).__init__()
        self.graph = graph
        self.contract_order = contract_order
        self.cores = []
        for item in range(len(graph)):
            shape = list(graph[:item, item]) + list(graph[item, item:])
            self.cores.append(torch.nn.Parameter(torch.randn([i if i!=0 else 1 for i in shape])))
            self.register_parameter('cores' + str(item), self.cores[-1])
        
        if self.contract_order:
            self.cores = [self.cores[i].permute(self.contract_order) for i in self.contract_order]

    def contract_graph(self):
        res = self.cores[0]
        N = len(self.cores)
        for i in range(1, N):
            shape_node = list(self.cores[i].shape)
            node = self.cores[i].reshape([np.prod(shape_node[:i])] + shape_node[i:])
            res = torch.tensordot(res, node, [[i], [0]])
            res = torch.movedim(res, N-1, i)
            shape = res.shape
            axis, new_shape = [k for k in range(i+1)], list(shape[:i+1])
            for j in range(N-i-1):
                axis.extend([i+1+j, N+j])
                new_shape.append(shape[i+1+j] * shape[N+j])
            res = res.permute(axis).reshape(new_shape)
        return res
        
    def forward(self):
        if self.contract_order:
            return self.contract_graph().permute([self.contract_order.index(i) for i in range(len(self.contract_order))])
        else:
            return self.contract_graph()


if __name__ == '__main__':
    graph = np.array([[3, 2, 0, 1, 0], [0, 2, 4, 2, 3], [0, 0, 4,0, 2],[0,0,0,2,2],[0,0,0,0,6]])
    net = TNGraph(graph, contract_order=[2,3,1,4,0])
    x = net().cuda()
    print(net.parameters())
    traget = torch.from_numpy(np.random.random_sample([3,2,4,2,6])).float()
    loss_function = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    for i in range(1):
        outputs = net()
        optimizer.zero_grad()
        loss = loss_function(outputs, traget)
        loss.backward()
        optimizer.step()

        if i % 1000==0:
            print(loss.item())


