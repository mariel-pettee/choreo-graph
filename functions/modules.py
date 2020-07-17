import numpy as np
import torch
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import MessagePassing, GatedGraphConv
from torch_geometric.utils import add_self_loops, degree
from torch.nn import Parameter, ModuleList
from torch.autograd import Variable
import torch.nn.functional as F
from .functions import *

class VAE(torch.nn.Module):
    """Graph Variational Autoencoder"""
    def __init__(self, node_features, edge_features, hidden_size, node_embedding_dim, edge_embedding_dim, input_size, output_size, num_layers):
        super(VAE, self).__init__()
        self.node_features = node_features
        self.node_embedding_dim = node_embedding_dim
        self.edge_features = edge_features
        self.edge_embedding_dim = edge_embedding_dim
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.encoder = MLPEncoder(
            node_features=self.node_features, 
            edge_features=self.edge_features, 
            hidden_size=self.hidden_size, 
            node_embedding_dim=self.node_embedding_dim,
            edge_embedding_dim=self.edge_embedding_dim)
        self.decoder = RNNDecoder(
            input_size=self.node_embedding_dim, 
            output_size=self.output_size,
            edge_embedding_dim=self.edge_embedding_dim,
            num_layers=self.num_layers)

    def forward(self, batch):
        node_embedding, edge_index, edge_embedding, log_probabilities = self.encoder(batch)
        z = torch.nn.functional.gumbel_softmax(log_probabilities, tau=0.5)
        output = self.decoder(node_embedding, edge_index, edge_embedding, z)
        return output

class MLPEncoder(torch.nn.Module):
    """Encoder for fully-connected graph inputs"""
    def __init__(self, node_features, edge_features, hidden_size, node_embedding_dim, edge_embedding_dim):
        super(MLPEncoder, self).__init__()
        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_size  = hidden_size
        self.node_embedding_dim = node_embedding_dim
        self.edge_embedding_dim = edge_embedding_dim
        self.node_embedding = Sequential(Linear(self.node_features, self.node_embedding_dim), ReLU())
#         self.node_embedding = Sequential(Linear(self.node_features, int(self.node_features/2)), ReLU(),Linear(int(self.node_features/2), int(self.node_features/3)), ReLU(),Linear(int(self.node_features/3), int(self.node_features/4)), ReLU(),Linear(int(self.node_features/4), int(self.node_features/5)), ReLU(),Linear(int(self.node_features/5), self.node_embedding_dim), ReLU())
        self.edge_embedding = Sequential(Linear(self.edge_features, self.edge_embedding_dim), ReLU())
        self.MLP = Sequential(Linear(self.edge_embedding_dim, self.hidden_size), 
                              ReLU(), 
                              Linear(self.hidden_size, 2*self.node_embedding_dim*self.node_embedding_dim))
        self.graph_conv_1 = MLPGraphConv(in_channels=self.node_embedding_dim, 
                                           out_channels=self.node_embedding_dim, 
                                           nn=self.MLP, root_weight=True, bias=True, aggr='add')
        self.graph_conv_2 = MLPGraphConv(in_channels=self.node_embedding_dim, 
                                           out_channels=self.node_embedding_dim, 
                                           nn=self.MLP, root_weight=True, bias=True, aggr='add')
        
        self.graph_conv_list = ModuleList([self.graph_conv_1, self.graph_conv_2])

    def forward(self, data):
        node_embedding = self.node_embedding(data.x)
        edge_embedding = self.edge_embedding(data.edge_attr)
        for layer in self.graph_conv_list:
            node_embedding = layer(node_embedding, data.edge_index, edge_embedding)
        return node_embedding, data.edge_index, edge_embedding, F.log_softmax(edge_embedding, dim=-1)

class RNNDecoder(torch.nn.Module):
    """Decoder from graph to predicted positions"""
    def __init__(self, input_size, output_size, edge_embedding_dim, num_layers):
        super(RNNDecoder, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.edge_embedding_dim = edge_embedding_dim
        self.num_layers = num_layers
        self.node_transform = Sequential(Linear(self.input_size, self.output_size), ReLU())
        self.graph_conv = GatedGraphConv(out_channels=self.output_size, num_layers=self.num_layers)

    def forward(self, x, edge_index, edge_attr, z):
        node_features = self.node_transform(x)
        edge_weight = torch.sum(z*edge_attr, axis=1)
        node_features = self.graph_conv(node_features, edge_index, edge_weight=edge_weight)
        return node_features
    
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

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)
        if self.root is not None:
            uniform(self.root.size(0), self.root)
        zeros(self.bias)
            
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

