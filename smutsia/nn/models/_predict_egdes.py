import torch
from torch_geometric.nn.conv import DynamicEdgeConv
from torch_geometric.nn.inits import glorot
from .. import MLP
from ..init import init_weights


class ClusterPrediction(torch.nn.Module):
    def __init__(self, in_channels: int):
        super(ClusterPrediction, self).__init__()
        self.in_channels = in_channels
        self.conv1 = DynamicEdgeConv(MLP([2 * in_channels, 64]), k=30)
        self.conv2 = DynamicEdgeConv(MLP([2 * 64, 64]), k=30)
        self.conv3 = DynamicEdgeConv(MLP([2 * 64, 64]), k=30)
        self.W = MLP([2 * 64, 64, 1], negative_slope=0.2, dropout=0.0)
        self.reset_parameters()

    def reset_parameters(self):
        init_weights(self.W, glorot)

    def forward(self, x, edge_index, batch=None):
        x = self.conv1(x, batch)
        x = self.conv2(x, batch)
        x = self.conv3(x, batch)
        values, indices = torch.sort(edge_index, dim=0)
        unique, inverse = torch.unique(values, return_inverse=True, dim=1)
        out = self.W(torch.cat([x[unique[0]], x[unique[1]]], dim=1))

        return torch.sigmoid(out[inverse]).view(-1)
