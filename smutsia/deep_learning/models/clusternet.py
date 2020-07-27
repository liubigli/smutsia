import torch
from torch import nn
from torch.nn import Sequential as Seq, Linear, ReLU, BatchNorm1d, Dropout, MaxPool2d
from torch_geometric.nn import EdgeConv, dense_mincut_pool, DynamicEdgeConv, global_max_pool
from torch_geometric.utils import to_dense_adj, to_dense_batch
from smutsia.point_cloud.repr import batch_knn_graph, torch_rri_representations

def MLP(channels):
    return Seq(*[
        Seq(Linear(channels[i - 1], channels[i]), ReLU(), BatchNorm1d(channels[i]))
        for i in range(1, len(channels))
    ])

class ClusterNet(nn.Module):
    def __init__(self, k, num_classes, rri_repr=False, aggr='max'):
        super(ClusterNet, self).__init__()
        self.k = k
        self.num_classes = num_classes
        self.rri_repr = rri_repr
        self.input_size = 64 if rri_repr else 3

        if self.rri_repr:
            self.conv0 = Linear(in_features=4, out_features=64)
            self.maxpool1 = MaxPool2d(kernel_size=(k, 1))

        self.conv1 = EdgeConv(MLP([2 * self.input_size, 64, 64, 64]), aggr=aggr)
        self.pool1 = Linear(in_features=64, out_features=32)
        self.conv2 = EdgeConv(MLP([2 * 64, 128]), aggr=aggr)
        self.pool2 = Linear(in_features=128, out_features=8)
        self.conv3 = EdgeConv(MLP([2 * 128, 256, 256]), aggr=aggr)
        self.conv4 = EdgeConv(MLP([2 * 256, 512, 512]), aggr=aggr)
        self.lin1 = MLP([256 + 512, 1024])
        self.final = MLP([1024, 512, 256, self.num_classes])

    def forward(self, x, batch=None):
        if self.rri_repr:
            rri, edge_index = torch_rri_representations(x, k=self.k, batch=batch)
            rri = rri.to(x.device)
            edge_index = edge_index.to(x.device)
            # RRI block input
            rri = self.conv0(rri)
            x = self.maxpool1(rri)
            x = x[..., 0, :]
        else:
            # first DGCNN
            edge_index = batch_knn_graph(x, k=self.k, batch=batch)

        x = self.conv1(x, edge_index)
        # first max cut pooling
        x, _ = to_dense_batch(x, batch=batch)
        adj1 = to_dense_adj(edge_index=edge_index, batch=batch)
        s1 = self.pool1(x)
        x, _, mc1, o1 = dense_mincut_pool(x, adj1, s1)

        # recompute batch for the pooled feature vector
        if len(x.size()) == 3:
            batch = torch.repeat_interleave(torch.arange(x.size(0)), x.size(1)).to(x.device)

        x = x.reshape(-1, x.size(-1))
        # second DGCNN
        edge_index = batch_knn_graph(x, k=self.k, batch=batch)
        x = self.conv2(x, edge_index)
        # second max cut pooling
        x, _ = to_dense_batch(x, batch=batch)
        adj2 = to_dense_adj(edge_index=edge_index, batch=batch)
        s2 = self.pool2(x)
        x, _, mc2, o2 = dense_mincut_pool(x, adj2, s2)

        # recompute batch for the pooled feature vector
        if len(x.size()) == 3:
            batch = torch.repeat_interleave(torch.arange(x.size(0)), x.size(1)).to(x.device)

        x = x.reshape(-1, x.size(-1))
        # third DGCNN
        edge_index = batch_knn_graph(x, k=self.k, batch=batch)
        x1 = self.conv3(x, edge_index)
        x2 = self.conv4(x1, edge_index)
        # concat third DGCNN
        out = self.lin1(torch.cat([x1, x2], dim=1))

        # global maxpool from 1024 to num_class features
        out = global_max_pool(out, batch)
        out = self.final(out)

        return nn.functional.log_softmax(out, dim=-1), mc1 + mc2, o1 + o2, (s1, s2)



class DCGNN(nn.Module):
    def __init__(self, k, num_classes, aggr='max'):
        super(DCGNN, self).__init__()
        self.k = k
        self.num_classes = num_classes
        self.conv1 = DynamicEdgeConv(MLP([2 * 3, 64, 64, 64]), k, aggr)
        self.conv2 = DynamicEdgeConv(MLP([2 * 64, 128]), k, aggr)
        self.lin1 = MLP([64 + 128, 1024])
        self.mlp = Seq(
            MLP([1024, 512]), Dropout(0.5), MLP([512, 256]), Dropout(0.5),
            Linear(256, num_classes))

    def forward(self, data):
        pos = data.pos
        batch = data.batch
        # edge_index = batch_knn_graph(x, k=self.k, batch=batch)
        x1 = self.conv1(pos, batch)
        x2 = self.conv2(x1, batch)
        out = self.lin1(torch.cat([x1, x2], dim=1))
        out = global_max_pool(out, batch)
        out = self.mlp(out)
        return nn.functional.log_softmax(out, dim=-1)