import numpy as np
import torch
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch.nn import Parameter

class MLPEncoder(torch.nn.Module):
    """Encoder for fully-connected graph inputs"""
    def __init__(self, node_features, edge_features, hidden_size, node_embedding_dim):
        super(MLPEncoder, self).__init__()
        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_size  = hidden_size
        self.node_embedding_dim = node_embedding_dim
        self.node_embedding = MLP(input_size = self.node_features, hidden_size = self.hidden_size, output_size = self.node_embedding_dim)
        self.edge_embedding = MLPGraphConv(in_channels=self.node_embedding_dim, out_channels=self.node_embedding_dim, nn=MLP(self.edge_features, self.hidden_size, 2*self.node_embedding_dim*self.node_embedding_dim))

    def forward(self, input):
        hidden = self.node_embedding(input)
        hidden = self.edge_embedding(hidden)
        return output
    
class MLPGraphConv(MessagePassing): # Heavily inspired by NNConv
    r"""
    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        nn (torch.nn.Module): A neural network that
            maps edge features :obj:`edge_attr` of shape :obj:`[-1,
            num_edge_features]` to shape
            :obj:`[-1, in_channels * out_channels]`, *e.g.*, defined by
            :class:`torch.nn.Sequential`.
        aggr (string, optional): The aggregation scheme to use
            (:obj:`"add"`, :obj:`"mean"`, :obj:`"max"`).
            (default: :obj:`"add"`)
        root_weight (bool, optional): If set to :obj:`False`, the layer will
            not add the transformed root node features to the output.
            (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 nn,
                 aggr='add',
                 root_weight=True,
                 bias=True,
                 **kwargs):
        super(MLPGraphConv, self).__init__(aggr=aggr, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nn = nn
        self.aggr = aggr

        if root_weight:
            self.root = Parameter(torch.Tensor(in_channels, out_channels))
        else:
            self.register_parameter('root', None)

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

    def forward(self, x, edge_index, edge_attr):
        """"""
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        pseudo = edge_attr.unsqueeze(-1) if edge_attr.dim() == 1 else edge_attr
        return self.propagate(edge_index, x=x, pseudo=pseudo)

    def message(self, x_i, x_j, pseudo):
        node_features = torch.tensor(torch.cat([x_i,x_j], dim=1))
        weight = self.nn(pseudo)
        weight = weight.view(node_features.shape[0], node_features.shape[1], self.out_channels)
        return torch.matmul(node_features.unsqueeze(1), weight).squeeze(1)

    def update(self, aggr_out, x):
        if self.root is not None:
            aggr_out = aggr_out + torch.mm(x, self.root)
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels, self.out_channels)

    
class NNConv(MessagePassing):
    r"""The continuous kernel-based convolutional operator from the
    `"Neural Message Passing for Quantum Chemistry"
    <https://arxiv.org/abs/1704.01212>`_ paper.
    This convolution is also known as the edge-conditioned convolution from the
    `"Dynamic Edge-Conditioned Filters in Convolutional Neural Networks on
    Graphs" <https://arxiv.org/abs/1704.02901>`_ paper (see
    :class:`torch_geometric.nn.conv.ECConv` for an alias):

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{\Theta} \mathbf{x}_i +
        \sum_{j \in \mathcal{N}(i)} \mathbf{x}_j \cdot
        h_{\mathbf{\Theta}}(\mathbf{e}_{i,j}),

    where :math:`h_{\mathbf{\Theta}}` denotes a neural network, *.i.e.*
    a MLP.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        nn (torch.nn.Module): A neural network :math:`h_{\mathbf{\Theta}}` that
            maps edge features :obj:`edge_attr` of shape :obj:`[-1,
            num_edge_features]` to shape
            :obj:`[-1, in_channels * out_channels]`, *e.g.*, defined by
            :class:`torch.nn.Sequential`.
        aggr (string, optional): The aggregation scheme to use
            (:obj:`"add"`, :obj:`"mean"`, :obj:`"max"`).
            (default: :obj:`"add"`)
        root_weight (bool, optional): If set to :obj:`False`, the layer will
            not add the transformed root node features to the output.
            (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 nn,
                 aggr='add',
                 root_weight=True,
                 bias=True,
                 **kwargs):
        super(NNConv, self).__init__(aggr=aggr, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nn = nn
        self.aggr = aggr

        if root_weight:
            self.root = Parameter(torch.Tensor(in_channels, out_channels))
        else:
            self.register_parameter('root', None)

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

#         self.reset_parameters()

#     def reset_parameters(self):
#         reset(self.nn)
#         uniform(self.in_channels, self.root)
#         uniform(self.in_channels, self.bias)


    def forward(self, x, edge_index, edge_attr):
        """"""
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        pseudo = edge_attr.unsqueeze(-1) if edge_attr.dim() == 1 else edge_attr
        return self.propagate(edge_index, x=x, pseudo=pseudo)

    def message(self, x_j, pseudo):
        weight = self.nn(pseudo).view(-1, self.in_channels, self.out_channels)
        return torch.matmul(x_j.unsqueeze(1), weight).squeeze(1)

    def update(self, aggr_out, x):
        if self.root is not None:
            aggr_out = aggr_out + torch.mm(x, self.root)
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)

class MLP(torch.nn.Module):
    """Fully-connected ReLU network with 1 hidden layer"""
    def __init__(self, input_size, hidden_size, output_size, activation=ReLU()):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.hidden_size  = hidden_size
        self.output_size = output_size
        self.layer_1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.layer_2 = torch.nn.Linear(self.hidden_size, self.output_size)
        self.activation = activation
    def forward(self, input):
        hidden = self.layer_1(input)
        hidden = self.activation(hidden)
        hidden = self.layer_2(hidden)
        output = self.activation(hidden)
        return output