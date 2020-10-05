import torch
from torch.nn import Module, Sequential as Seq, Dropout, Linear, functional as F
from torch_geometric.nn import DynamicEdgeConv

from ._point_net import TransformNet
from .. import MLP

class DGCNN(Module):
    def __init__(self, k, num_classes, hidden_feat=64, dropout=0.5, aggr='max', transform=False):
        super(DGCNN, self).__init__()
        self.k = k
        self.num_classes = num_classes
        self.transform = transform

        if transform:
            self.tnet = TransformNet()

            self.conv1 = DynamicEdgeConv(
                nn=MLP([2 * 3, hidden_feat, hidden_feat], bias=False, negative_slope=0.2),
                k=k,
                aggr=aggr
            )
            self.conv2 = DynamicEdgeConv(
                nn=MLP([2 * hidden_feat, hidden_feat, hidden_feat], bias=False, negative_slope=0.2),
                k=k,
                aggr=aggr
            )
            self.conv3 = DynamicEdgeConv(
                nn=MLP([2 * hidden_feat, hidden_feat, hidden_feat], bias=False, negative_slope=0.2),
                k=k,
                aggr=aggr
            )
            self.lin1 = MLP([3 * hidden_feat, 1024], bias=False, negative_slope=0.2)

            self.mlp = Seq(
                MLP([1024, 256], bias=False, negative_slope=0.2),
                Dropout(dropout),
                MLP([256, 128], bias=False, negative_slope=0.2),
                Dropout(dropout),
                Linear(128, num_classes, bias=False)
            )

        else:
            self.conv1 = DynamicEdgeConv(MLP([2 * 3, hidden_feat]), k, aggr)
            self.conv2 = DynamicEdgeConv(MLP([2 * hidden_feat, hidden_feat]), k, aggr)
            self.conv3 = DynamicEdgeConv(MLP([2 * hidden_feat, hidden_feat]), k, aggr)

            self.lin1 = MLP([3 * hidden_feat, 1024])

            self.mlp = Seq(MLP([1024, 256]), Dropout(dropout), MLP([256, 128]),
                       Dropout(dropout), Linear(128, num_classes))

    def forward(self, x, batch=None):
        if self.transform:
            tr = self.tnet(x, batch=batch)

            if batch is None:
                x = torch.matmul(x, tr[0])
            else:
                batch_size = batch.max().item() + 1
                x = torch.cat([torch.matmul(x[batch==i], tr[i]) for i in range(batch_size)])

        x1 = self.conv1(x, batch)
        x2 = self.conv2(x1, batch)
        x3 = self.conv3(x2, batch)
        out = self.lin1(torch.cat([x1, x2, x3], dim=1))
        out = self.mlp(out)

        return F.log_softmax(out, dim=-1)
