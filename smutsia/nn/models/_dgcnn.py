import torch
from torch.nn import Module, Sequential as Seq, Dropout, Linear, functional as F
from torch_geometric.nn import DynamicEdgeConv

from .. import MLP

class DGCNN(Module):
    def __init__(self, k, num_classes, hidden_feat=64, dropout=0.5, aggr='max'):
        super(DGCNN, self).__super__()
        self.k = k
        self.num_classes = num_classes

        self.conv1 = DynamicEdgeConv(MLP([2 * 3, hidden_feat]), k, aggr)
        self.conv2 = DynamicEdgeConv(MLP([2 * hidden_feat, hidden_feat]), k, aggr)
        self.conv3 = DynamicEdgeConv(MLP([2 * hidden_feat, hidden_feat]), k, aggr)

        self.lin1 = MLP([3 * hidden_feat, 1024])

        self.mlp = Seq(MLP([1024, 256]), Dropout(dropout), MLP([256, 128]),
                       Dropout(dropout), Linear(128, num_classes))

    def forward(self, x, batch=None):
        x1 = self.conv1(x, batch)
        x2 = self.conv2(x1, batch)
        x3 = self.conv3(x2, batch)
        out = self.lin1(torch.cat([x1, x2, x3], dim=1))
        out = self.mlp(out)

        return F.log_softmax(out, dim=-1)
