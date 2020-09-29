import torch
from torch.nn import Sequential as Seq, Linear, Dropout, Module
from torch.nn import functional as F
from smutsia.nn.layers.morpho_layers import DilateEdgeConv, DilateFlatEdgeConv, ErodeEdgeConv
from .. import MLP


class DilateDGNN(Module):
    def __init__(self, k, num_classes):
        super(DilateDGNN, self).__init__()
        self.k = k
        self.num_classes = num_classes

        # self.conv1 = DilateEdgeConv(in_channels=3, nb_filters=20, k=k)
        # self.conv2 = DilateEdgeConv(in_channels=60, nb_filters=2, k=k)
        # self.conv3 = DilateEdgeConv(in_channels=120, nb_filters=2, k=k)
        #
        # self.lin1 = MLP([420, 1024])

        self.conv1 = DilateEdgeConv(in_channels=3, nb_filters=3, k=k)
        self.conv2 = DilateEdgeConv(in_channels=9, nb_filters=3, k=k)
        self.conv3 = DilateEdgeConv(in_channels=27, nb_filters=3, k=k)

        self.lin1 = MLP([117, 1024])

        self.mlp = Seq(MLP([1024, 256]), Dropout(0.5), MLP([256, 128]),
                       Dropout(0.5), Linear(128, num_classes))

    def forward(self, x, batch=None):
        x1 = self.conv1(x, batch)
        x2 = self.conv2(x1, batch)
        x3 = self.conv3(x2, batch)

        out = self.lin1(torch.cat([x1, x2, x3], dim=1))

        out = self.mlp(out)
        return F.log_softmax(out, dim=-1)


class ErodeDGNN(Module):
    def __init__(self, k, num_classes):
        super(ErodeDGNN, self).__init__()
        self.k = k
        self.num_classes = num_classes

        self.conv1 = ErodeEdgeConv(in_channels=3, out_channels=64, k=k)
        self.conv2 = ErodeEdgeConv(in_channels=64, out_channels=64, k=k)
        self.conv3 = ErodeEdgeConv(in_channels=64, out_channels=64, k=k)

        self.lin1 = MLP([3 * 64, 1024])

        self.mlp = Seq(MLP([1024, 256]), Dropout(0.5), MLP([256, 128]),
                       Dropout(0.5), Linear(128, num_classes))

    def forward(self, x, batch=None):
        x1 = self.conv1(x, batch)
        x2 = self.conv2(x1, batch)
        x3 = self.conv3(x2, batch)

        out = self.lin1(torch.cat([x1, x2, x3], dim=1))

        out = self.mlp(out)
        return F.log_softmax(out, dim=-1)


class MorphoGradDGNN(Module):
    def __init__(self, k, num_classes):
        super(MorphoGradDGNN, self).__init__()
        self.k = k
        self.num_classes = num_classes

        self.conv_dil1 = DilateFlatEdgeConv(in_channels=3, out_channels=64, k=k)
        self.conv_ero1 = ErodeEdgeConv(in_channels=3, out_channels=64, k=k)

        self.conv_dil2 = DilateFlatEdgeConv(in_channels=64, out_channels=64, k=k)
        self.conv_ero2 = ErodeEdgeConv(in_channels=64, out_channels=64, k=k)

        self.conv_dil3 = DilateFlatEdgeConv(in_channels=64, out_channels=64, k=k)
        self.conv_ero3 = ErodeEdgeConv(in_channels=64, out_channels=64, k=k)

        self.lin1 = MLP([3 * 64, 1024])

        self.mlp = Seq(MLP([1024, 256]), Dropout(0.5), MLP([256, 128]),
                       Dropout(0.5), Linear(128, num_classes))

    def forward(self, x, batch=None):
        xdil1 = self.conv_dil1(x, batch)
        xero1 = self.conv_ero1(x, batch)
        x1 = xdil1 - xero1

        xdil2 = self.conv_dil2(x1, batch)
        xero2 = self.conv_ero2(x1, batch)
        x2 = xdil2 - xero2

        xdil3 = self.conv_dil2(x2, batch)
        xero3 = self.conv_ero2(x2, batch)
        x3 = xdil3 - xero3

        out = self.lin1(torch.cat([x1, x2, x3], dim=1))

        out = self.mlp(out)
        return F.log_softmax(out, dim=-1)
