from typing import Optional, List
import torch
from torch import nn
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d, LayerNorm, InstanceNorm1d, Dropout, MaxPool2d
from torch_geometric.nn import EdgeConv, dense_mincut_pool, knn_graph
from torch_geometric.utils import to_dense_adj, to_dense_batch
from smutsia.point_cloud.repr import torch_rri_representations
from smutsia.point_cloud.repr import batch_knn_graph

class MLP(Sequential):
    def __init__(self, channels: List[int], norm: Optional[str] = None,
                 bias: bool = True, dropout: float = 0.):
        m = []
        for i in range(1, len(channels)):
            m.append(Linear(channels[i - 1], channels[i], bias))

            if i < len(channels) - 1:
                if norm and norm == 'batch':
                    m.append(BatchNorm1d(channels[i], affine=True))
                elif norm and norm == 'layer':
                    m.append(LayerNorm(channels[i], elementwise_affine=True))
                elif norm and norm == 'instance':
                    m.append(InstanceNorm1d(channels[i], affine=False))
                elif norm:
                    raise NotImplementedError(
                        f'Normalization layer "{norm}" not supported.')
                m.append(ReLU())
                m.append(Dropout(dropout))

        super(MLP, self).__init__(*m)


class ClusterNet(nn.Module):
    def __init__(self, k, num_classes, aggr='max'):
        super(ClusterNet, self).__init__()
        self.k = k
        self.num_classes = num_classes
        self.rri_conv1 = Linear(in_features=4, out_features=64, bias=False)
        self.maxpool1 = MaxPool2d(kernel_size=(k, 1))
        self.conv1 = EdgeConv(MLP([2 * 64, 128]), aggr=aggr)
        self.pool1 = Linear(in_features=128, out_features=32)
        self.conv2 = EdgeConv(MLP([2 * 128, 256]), aggr=aggr)
        self.pool2 = Linear(in_features=256, out_features=8)
        self.conv3 = EdgeConv(MLP([2 * 256, 1024]))
        self.pool3 = Linear(in_features=1024, out_features=1)
        self.final = MLP([1024, 512, 256, self.num_classes])

    def forward(self, x, batch=None):
        if len(x.size()) == 3:
            batch = torch.repeat_interleave(torch.arange(x.size(0)), x.size(1))

        rri, edge_index = torch_rri_representations(x, self.k, batch=batch)
        rri = self.rri_conv1(rri)
        x = self.maxpool1(rri)
        x = x[..., 0, :]
        # first DGCNN
        x = self.conv1(x, edge_index)
        # first max cut pooling
        x, _ = to_dense_batch(x, batch=batch)
        adj = to_dense_adj(edge_index=edge_index, batch=batch)
        s = self.pool1(x)
        x, _, mc1, o1 = dense_mincut_pool(x, adj, s)

        # recompute batch for the pooled feature vector
        if len(x.size()) == 3:
            batch = torch.repeat_interleave(torch.arange(x.size(0)), x.size(1))

        x = x.reshape(-1, x.size(-1))
        # second DGCNN
        edge_index = batch_knn_graph(x, k=self.k, batch=batch)
        x = self.conv2(x, edge_index)

        # second max cut pooling
        x, mask = to_dense_batch(x, batch=batch)
        adj = to_dense_adj(edge_index=edge_index, batch=batch)
        s = self.pool2(x)
        x, _, mc2, o2 = dense_mincut_pool(x, adj, s)

        # recompute batch for the pooled feature vector
        if len(x.size()) == 3:
            batch = torch.repeat_interleave(torch.arange(x.size(0)), x.size(1))
        x = x.reshape(-1, x.size(-1))
        # third DGCNN
        edge_index = batch_knn_graph(x, k=self.k, batch=batch)
        x = self.conv3(x, edge_index)

        # third max cut pooling
        x, mask = to_dense_batch(x, batch=batch)
        adj = to_dense_adj(edge_index=edge_index, batch=batch)
        s = self.pool3(x)
        x, _, mc3, o3 = dense_mincut_pool(x, adj, s)

        # final flatten and MLP
        x = self.final(x)
        if self.num_classes == 1:
            x = torch.sigmoid(x)
        else:
            x = torch.softmax(x, dim=-1)

        return x[..., 0, :], mc1 + mc2 + mc3, o1 + o2 + o3
