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
    def __init__(self, node_features, edge_features, hidden_size, node_embedding_dim, edge_embedding_dim, input_size, output_size, num_layers, sampling, recurrent):
        super(VAE, self).__init__()
        self.node_features = node_features
        self.node_embedding_dim = node_embedding_dim
        self.edge_features = edge_features
        self.edge_embedding_dim = edge_embedding_dim
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.sampling = sampling
        self.recurrent = recurrent
        self.encoder = MLPEncoder(
            node_features=self.node_features, 
            edge_features=self.edge_features, 
            hidden_size=self.hidden_size, 
            node_embedding_dim=self.node_embedding_dim,
            edge_embedding_dim=self.edge_embedding_dim)
        self.decoder = Decoder(
            input_size=self.node_embedding_dim, 
            output_size=self.output_size,
            num_layers=self.num_layers,
            edge_embedding_dim=self.edge_embedding_dim,
            hidden_size=self.hidden_size,
            sampling=self.sampling,
            recurrent=self.recurrent,
        )

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
        self.edge_embedding = Sequential(Linear(self.edge_features, self.edge_embedding_dim), ReLU())
        self.MLP = Sequential(Linear(self.edge_features, self.hidden_size), 
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
        for layer in self.graph_conv_list:
            node_embedding = layer(node_embedding, data.edge_index, data.edge_attr)
        edge_embedding = self.edge_embedding(data.edge_attr)
        return node_embedding, data.edge_index, edge_embedding, F.log_softmax(edge_embedding, dim=-1)

class NRIEncoder(torch.nn.Module):
    """Encoder for fully-connected graph inputs"""
    def __init__(self, node_features, edge_features, hidden_size, node_embedding_dim, edge_embedding_dim):
        super(NRIEncoder, self).__init__()
        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_size  = hidden_size
        self.node_embedding_dim = node_embedding_dim
        self.edge_embedding_dim = edge_embedding_dim
        self.node_embedding = Sequential(Linear(self.node_features, self.node_embedding_dim), ReLU())
        self.edge_embedding = Sequential(Linear(self.edge_features, self.edge_embedding_dim), ReLU())
        self.MLP = Sequential(Linear(self.edge_features, self.hidden_size), 
                              ReLU(), 
                              Linear(self.hidden_size, 2*self.node_embedding_dim*self.node_embedding_dim))
        self.MLP_2 = Sequential(Linear(self.edge_features, self.hidden_size), 
                      ReLU(), 
                      Linear(self.hidden_size, 2*self.node_embedding_dim*self.node_embedding_dim))
        self.MLP_3 = Sequential(Linear(self.edge_features, self.hidden_size), 
                  ReLU(), 
                  Linear(self.hidden_size, 2*self.node_embedding_dim*self.node_embedding_dim))
        self.graph_conv = MLPGraphConv(in_channels=self.node_embedding_dim, 
                                           out_channels=self.node_embedding_dim, 
                                           nn=self.MLP, nn_2=self.MLP_2, root_weight=True, bias=True, aggr='add')

    def forward(self, data):
        node_embedding = self.node_embedding(data.x)
        node_embedding = self.graph_conv(node_embedding, data.edge_index)
#         node_skip = node_embedding
        edge_embedding = self.edge_embedding(data.edge_attr)
        node_embedding = self.MLP_3(node_embedding) # this actually needs the concatenation of edge attributes (cartesian product)
        return node_embedding, data.edge_index, edge_embedding, F.log_softmax(node_embedding, dim=-1)
    
    
class Decoder(torch.nn.Module):
    """Decoder from graph to predicted positions"""
    def __init__(self, input_size, hidden_size, output_size, num_layers, edge_embedding_dim, sampling, recurrent):
        super(Decoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.edge_embedding_dim = edge_embedding_dim
        self.sampling = sampling
        self.recurrent = recurrent
        self.node_transform = Sequential(Linear(self.input_size, self.output_size), ReLU())
        if self.recurrent == True:
            self.rnn_graph_conv = GatedGraphConv(out_channels=self.input_size, num_layers=self.num_layers)
        else: 
            self.MLP = Sequential(Linear(self.edge_embedding_dim, self.hidden_size), 
                                  ReLU(), 
                                  Linear(self.hidden_size, 2*self.input_size*self.input_size))
            self.graph_conv_1 = MLPGraphConv(in_channels=self.input_size, 
                                               out_channels=self.input_size, 
                                               nn=self.MLP, root_weight=True, bias=True, aggr='add')
            self.graph_conv_2 = MLPGraphConv(in_channels=self.input_size, 
                                               out_channels=self.input_size, 
                                               nn=self.MLP, root_weight=True, bias=True, aggr='add')

            self.graph_conv_list = ModuleList([self.graph_conv_1, self.graph_conv_2])

    def forward(self, x, edge_index, edge_attr, z):
        
        if self.sampling == True: 
            edge_weight = torch.sum(z*edge_attr, axis=1)
        else: 
            edge_weight = None
        
        if self.recurrent == True: 
            # use GatedGraphConv w/ RNN
            x = self.rnn_graph_conv(x, edge_index, edge_weight=edge_weight)
        else: 
            # use MLPGraphConv layers
            for layer in self.graph_conv_list:
                x = layer(x, edge_index, edge_attr)
        
        node_features = self.node_transform(x) # transform into real coordinates
        return node_features

    
class NRIGraphConv(MessagePassing): # Heavily inspired by NNConv
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
                 nn_2,
                 aggr='add',
                 root_weight=True,
                 bias=True,
                 **kwargs):
        super(NRIGraphConv, self).__init__(aggr=aggr, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nn = nn
        self.nn_2 = nn_2
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
        reset(self.nn_2)
        if self.root is not None:
            uniform(self.root.size(0), self.root)
        zeros(self.bias)
            
    def forward(self, x, edge_index):
        """"""
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        node_features = torch.tensor(torch.cat([x_i,x_j], dim=1))
        return self.nn(node_features)

    def update(self, aggr_out, x):
        aggr_out = self.nn_2(aggr_out) 

        # Potentially use x to do skip_connection here 
        
#         if self.root is not None:
#             aggr_out = aggr_out + torch.mm(x, self.root)
#         if self.bias is not None:
#             aggr_out = aggr_out + self.bias

        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels, self.out_channels)


    
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

