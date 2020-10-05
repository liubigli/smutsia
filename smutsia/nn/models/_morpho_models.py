import torch
from torch.nn import Sequential as Seq, Linear, Dropout, Module
from torch.nn import functional as F
from torch_geometric.nn import DynamicEdgeConv
from smutsia.nn.layers.morpho_layers import DilateEdgeConv, DilateMaxPlus, DilateFlatEdgeConv, ErodeEdgeConv, Delirium

from .. import MLP


class DilateDGNN(Module):
    def __init__(self, k, num_classes, in_channels=3, nb_filters=20, hidden_features=64, kind='edge-conv'):
        super(DilateDGNN, self).__init__()
        self.k = k
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.nb_filters = nb_filters
        self.kind = kind

        if kind == 'flat':
            self.hidden_features = hidden_features
            self.conv1 = DilateFlatEdgeConv(in_channels=self.in_channels, out_channels=self.hidden_features, k=k)
            self.conv2 = DilateFlatEdgeConv(in_channels=self.hidden_features, out_channels=self.hidden_features, k=k)
            self.conv3 = DilateFlatEdgeConv(in_channels=self.hidden_features, out_channels=self.hidden_features, k=k)

            self.lin1 = MLP([3 * self.hidden_features, 1024])
        elif kind == 'max-plus':
            self.hidden_features = hidden_features
            self.conv1 = DilateMaxPlus(in_channels=self.in_channels, out_channels=self.hidden_features, k=k)
            self.conv2 = DilateMaxPlus(in_channels=self.hidden_features, out_channels=self.hidden_features, k=k)
            self.conv3 = DilateMaxPlus(in_channels=self.hidden_features, out_channels=self.hidden_features, k=k)

            self.lin1 = MLP([3 * self.hidden_features, 1024])
        else:
            self.conv1 = DilateEdgeConv(in_channels=self.in_channels, nb_filters=self.nb_filters, k=k)
            self.conv2 = DilateEdgeConv(in_channels=self.in_channels*self.nb_filters, nb_filters=2, k=k)
            self.conv3 = DilateEdgeConv(in_channels=self.in_channels*self.nb_filters*2, nb_filters=2, k=k)

            lin_in_channel = self.in_channels * self.nb_filters * 7  # hard coded
            self.lin1 = MLP([lin_in_channel, 1024])

        self.mlp = Seq(MLP([1024, 256]), Dropout(0.5), MLP([256, 128]),
                       Dropout(0.5), Linear(128, num_classes))

    def forward(self, x, batch=None):
        x1 = self.conv1(x, batch)
        x2 = self.conv2(x1, batch)
        x3 = self.conv3(x2, batch)

        out = self.lin1(torch.cat([x1, x2, x3], dim=1))

        out = self.mlp(out)
        return F.log_softmax(out, dim=-1)


class HybridDGNN(Module):
    def __init__(self, k, num_classes, in_channels=3, nb_filters=10, out_channels=64):
        super(HybridDGNN, self).__init__()
        self.k = k
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.nb_filters = nb_filters
        self.out_channels = out_channels

        self.conv1 = Delirium(in_channels=self.in_channels, out_channels=self.out_channels, k=k)
        self.conv2 = Delirium(in_channels=self.out_channels, out_channels=self.out_channels, k=k)
        self.conv3 = Delirium(in_channels=self.out_channels, out_channels=self.out_channels, k=k)

        lin_in_channel = 3 * self.out_channels
        self.lin1 = MLP([lin_in_channel, 1024])

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
    def __init__(self, k, num_classes, in_channels=3, nb_filters=20, flat=False, hidden_features=64):
        super(ErodeDGNN, self).__init__()
        self.k = k
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.nb_filters = nb_filters
        self.flat = flat
        if flat:
            self.hidden_features = hidden_features
            self.conv1 = DilateFlatEdgeConv(in_channels=self.in_channels, out_channels=self.hidden_features, k=k)
            self.conv2 = DilateFlatEdgeConv(in_channels=self.hidden_features, out_channels=self.hidden_features, k=k)
            self.conv3 = DilateFlatEdgeConv(in_channels=self.hidden_features, out_channels=self.hidden_features, k=k)

            self.lin1 = MLP([3 * 64, 1024])
        else:
            self.conv1 = ErodeEdgeConv(in_channels=self.in_channels, nb_filters=self.nb_filters, k=k)
            self.conv2 = ErodeEdgeConv(in_channels=self.in_channels*self.nb_filters, nb_filters=2, k=k)
            self.conv3 = ErodeEdgeConv(in_channels=self.in_channels*self.nb_filters*2, nb_filters=2, k=k)

            lin_in_channel = self.in_channels * sum([self.nb_filters ** i for i in range(1, 4)])
            self.lin1 = MLP([lin_in_channel, 1024])

        self.mlp = Seq(MLP([1024, 256]), Dropout(0.5), MLP([256, 128]),
                       Dropout(0.5), Linear(128, num_classes))

    def forward(self, x, batch=None):
        if self.flat:
            x1 = -self.conv1(-x, batch)
            x2 = -self.conv1(-x1, batch)
            x3 = -self.conv1(-x2, batch)
        else:
            x1 = self.conv1(x, batch)
            x2 = self.conv2(x1, batch)
            x3 = self.conv3(x2, batch)

        out = self.lin1(torch.cat([x1, x2, x3], dim=1))

        out = self.mlp(out)
        return F.log_softmax(out, dim=-1)


class MorphoGradDGNN(Module):
    def __init__(self, k, num_classes, in_channels=3, nb_filters=6, out_channels=64):
        super(MorphoGradDGNN, self).__init__()
        self.k = k
        self.num_classes = num_classes
        self.nb_filters = nb_filters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = DynamicEdgeConv(MLP([2 * 3, self.out_channels]), k=k)
        self.dil1 = DilateFlatEdgeConv(in_channels=self.out_channels, out_channels=self.out_channels, k=k)
        # self.ero1 = DilateFlatEdgeConv(in_channels=self.out_channels, out_channels=self.out_channels, k=k)
        self.dil2 = DilateFlatEdgeConv(in_channels=self.out_channels, out_channels=self.out_channels, k=k)
        # self.ero2 = DilateFlatEdgeConv(in_channels=self.out_channels, out_channels=self.out_channels, k=k)
        self.dil3 = DilateFlatEdgeConv(in_channels=self.out_channels, out_channels=self.out_channels, k=k)
        # self.ero3 = DilateFlatEdgeConv(in_channels=self.out_channels, out_channels=self.out_channels, k=k)
        # lin_in_channel = self.in_channels * sum([self.nb_filters ** i for i in range(1,4)])
        # lin_in_channel = self.in_channels * sum([self.nb_filters ** i for i in range(1,4)])
        # self.lin1 = MLP([lin_in_channel, 1024])
        self.lin1 = MLP([3 * self.out_channels, 1024])

        self.mlp = Seq(MLP([1024, 256]), Dropout(0.5), MLP([256, 128]),
                       Dropout(0.5), Linear(128, num_classes))

    def forward(self, x, batch=None):
        x_feat = self.conv1(x, batch)
        xdil1 = self.dil1(x_feat, batch)
        # xero1 = self.ero1(-x, batch)
        x1 = xdil1 - x_feat

        xdil2 = self.dil2(x1, batch)
        # xero2 = self.ero2(-x1, batch)
        x2 = xdil2 - x1

        xdil3 = self.dil3(x2, batch)
        # xero3 = self.ero3(-x2, batch)
        x3 = xdil3 - x2

        out = self.lin1(torch.cat([x1, x2, x3], dim=1))

        out = self.mlp(out)
        return F.log_softmax(out, dim=-1)
