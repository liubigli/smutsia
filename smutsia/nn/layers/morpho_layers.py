import torch
from torch.nn import Parameter
from torch.nn.functional import relu
from torch_scatter import scatter_add, scatter, segment_csr
from torch_geometric.nn import MessagePassing, knn_graph
from torch_geometric.nn.inits import normal, zeros
from torch_geometric.utils import add_remaining_self_loops


def dot_operation(x, weights):
    """
    In the (max, + ) algebra dot product is replaced by sum.
    This function implements the dot product between an input tensor x of shape (N, d)
    and a weight tensor of shape (d, h)
    """
    return torch.stack([torch.max(xel - relu(weights.T), dim=1)[0] for xel in x])


class DilateEdgeConv(MessagePassing):
    def __init__(self, in_channels: int, nb_filters: int, k: int, flat_kernel: bool = False,
                 bias: bool = False, add_self_loops: bool = False):

        super(DilateEdgeConv, self).__init__(aggr='max')
        self.k = k
        self.in_channels = in_channels
        self.nb_filters = nb_filters
        self.flat_kernels = flat_kernel
        self.add_self_loops = add_self_loops

        self.weights = Parameter(torch.Tensor(k, nb_filters), requires_grad=True)

        if bias:
            self.bias = Parameter(torch.Tensor(in_channels * nb_filters), requires_grad=True)
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        # TODO: ask Santiago about weight initialization
        normal(self.weights, 2, 1)
        if self.bias is not None:
            zeros(self.bias)

    def forward(self, x, batch=None):
        edge_index = knn_graph(x, k=self.k, batch=batch, loop=self.add_self_loops)
        out = self.propagate(edge_index=edge_index, x=x, size=None)

        if self.bias is not None:
            out = torch.max(out, self.bias)

        return out

    def message(self, x_j):
        ### for the moment this implementation works only with two dimensional tensors
        x_ik = x_j.reshape(-1, self.k, x_j.shape[-1])
        x_out = x_ik.repeat_interleave(self.nb_filters, dim=-1)
        x_out = x_out - relu(self.weights.repeat_interleave(x_j.shape[-1], dim=-1))

        return x_out.reshape(-1, self.nb_filters * x_j.shape[-1])

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.nb_filters)


class DilateFlatEdgeConv(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int, k: int, bias: bool = False, add_self_loops: bool = False):
        super(DilateFlatEdgeConv, self).__init__(aggr='max')
        self.k = k
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.add_self_loops = add_self_loops
        self.weights = Parameter(torch.Tensor(in_channels, out_channels), requires_grad=True)
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels), requires_grad=True)
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        # TODO: ask Santiago about weight initialization
        normal(self.weights, 2, 1)
        if self.bias is not None:
            zeros(self.bias)

    def forward(self, x, batch=None):
        edge_index = knn_graph(x, k=self.k, batch=batch, loop=self.add_self_loops)
        op = torch.stack([torch.max(xel - relu(self.weights.T), dim=1)[0] for xel in x])

        out = self.propagate(edge_index=edge_index, x=op, size=None)

        if self.bias is not None:
            out = torch.max(out, self.bias)

        return out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class ErodeEdgeConv(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int, k: int, bias: bool = False, add_self_loops: bool = False):
        super(ErodeEdgeConv, self).__init__()
        self.aggr = "min"
        self.k = k
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.add_self_loops = add_self_loops
        self.weights = Parameter(torch.Tensor(in_channels, out_channels), requires_grad=True)
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels), requires_grad=True)
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        # TODO: ask Santiago about weight initialization
        normal(self.weights, 2, 1)
        if self.bias is not None:
            zeros(self.bias)

    def forward(self, x, batch=None):
        edge_index = knn_graph(x, k=self.k, batch=batch, loop=self.add_self_loops)
        op = torch.stack([torch.max(xel + relu(self.weights.T), dim=1)[0] for xel in x])

        out = self.propagate(edge_index=edge_index, x=op, size=None)

        if self.bias is not None:
            out = torch.max(out, self.bias)

        return out

    def aggregate(self, inputs, index, ptr=None, dim_size=None):
        r"""Aggregates messages from neighbors as
        :math:`\square_{j \in \mathcal{N}(i)}`.

        Takes in the output of message computation as first argument and any
        argument which was initially passed to :meth:`propagate`.

        By default, this function will delegate its call to scatter functions
        that support "add", "mean" and "max" operations as specified in
        :meth:`__init__` by the :obj:`aggr` argument.
        """

        if ptr is not None:
            for _ in range(self.node_dim):
                ptr = ptr.unsqueeze(0)
            return segment_csr(inputs, ptr, reduce=self.aggr)
        else:
            return scatter(inputs, index, dim=self.node_dim, dim_size=dim_size,
                           reduce=self.aggr)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class GradEdgeConv(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int, k: int, bias: bool = False, add_self_loops: bool = False):
        super(GradEdgeConv, self).__init__()

        self.k = k
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.add_self_loops = add_self_loops
        self.weights = Parameter(torch.Tensor(in_channels, out_channels), requires_grad=True)
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels), requires_grad=True)
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        # TODO: ask Santiago about weight initialization
        normal(self.weights, 2, 1)
        if self.bias is not None:
            zeros(self.bias)

    def forward(self, x, batch=None):
        edge_index = knn_graph(x, k=self.k, batch=batch, loop=self.add_self_loops)
        op = torch.stack([torch.max(xel + relu(self.weights.T), dim=1)[0] for xel in x])

        out = self.propagate(edge_index=edge_index, x=op, size=None)

        if self.bias is not None:
            out = torch.max(out, self.bias)

        return out

    def aggregate(self, inputs, index, ptr=None, dim_size=None):
        r"""Aggregates messages from neighbors as
        :math:`\square_{j \in \mathcal{N}(i)}`.

        Takes in the output of message computation as first argument and any
        argument which was initially passed to :meth:`propagate`.

        By default, this function will delegate its call to scatter functions
        that support "add", "mean" and "max" operations as specified in
        :meth:`__init__` by the :obj:`aggr` argument.
        """

        if ptr is not None:
            for _ in range(self.node_dim):
                ptr = ptr.unsqueeze(0)
            return segment_csr(inputs, ptr, reduce=self.aggr)
        else:
            return scatter(inputs, index, dim=self.node_dim, dim_size=dim_size,
                           reduce=self.aggr)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class DilateConv(MessagePassing):
    r"""The graph convolutional operator from the `"Semi-supervised
    Classification with Graph Convolutional Networks"
    <https://arxiv.org/abs/1609.02907>`_ paper

    .. math::
        \mathbf{X}^{\prime} = \mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
        \mathbf{\hat{D}}^{-1/2} \mathbf{X} \mathbf{\Theta},

    where :math:`\mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}` denotes the
    adjacency matrix with inserted self-loops and
    :math:`\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij}` its diagonal degree matrix.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        improved (bool, optional): If set to :obj:`True`, the layer computes
            :math:`\mathbf{\hat{A}}` as :math:`\mathbf{A} + 2\mathbf{I}`.
            (default: :obj:`False`)
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the computation of :math:`\mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
            \mathbf{\hat{D}}^{-1/2}` on first execution, and will use the
            cached version for further executions.
            This parameter should only be set to :obj:`True` in transductive
            learning scenarios. (default: :obj:`False`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        normalize (bool, optional): Whether to add self-loops and apply
            symmetric normalization. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(self, in_channels, out_channels, improved=False, cached=False,
                 bias=True, normalize=True, **kwargs):
        super(DilateConv, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.normalize = normalize

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        normal(self.weight, mean=2, std=1)
        zeros(self.bias)
        self.cached_result = None
        self.cached_num_edges = None

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight=None, improved=False,
             dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)

        fill_value = 1 if not improved else 2
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_weight=None):
        """"""
        x = dot_operation(x, self.weight)

        if self.cached and self.cached_result is not None:
            if edge_index.size(1) != self.cached_num_edges:
                raise RuntimeError(
                    'Cached {} number of edges, but found {}. Please '
                    'disable the caching behavior of this layer by removing '
                    'the `cached=True` argument in its constructor.'.format(
                        self.cached_num_edges, edge_index.size(1)))

        if not self.cached or self.cached_result is None:
            self.cached_num_edges = edge_index.size(1)
            if self.normalize:
                edge_index, norm = self.norm(edge_index, x.size(
                    self.node_dim), edge_weight, self.improved, x.dtype)
            else:
                norm = edge_weight
            self.cached_result = edge_index, norm

        edge_index, norm = self.cached_result

        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        ## TODO Check if the * should be replaced by a +
        return norm.view(-1, 1) * x_j if norm is not None else x_j

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = torch.max(aggr_out , self.bias)
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)