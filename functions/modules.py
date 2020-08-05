import numpy as np
import torch
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import MessagePassing, GatedGraphConv
from torch_geometric.utils import add_self_loops, degree
from torch.nn import Parameter, ModuleList
from torch.autograd import Variable
import torch.nn.functional as F
from .functions import *
from torch_geometric.data import Data
# import pdb; pdb.set_trace()

#### UMBRELLA MODELS

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
    
class NRI(torch.nn.Module):
    """Implementation of NRI with Pytorch Geometric"""
    def __init__(self, node_features, edge_features, hidden_size, skip_connection, node_embedding_dim, edge_embedding_dim, dynamic_graph, seq_len):
        super(NRI, self).__init__()
        self.node_features = node_features
        self.node_embedding_dim = node_embedding_dim
        self.edge_features = edge_features
        self.edge_embedding_dim = edge_embedding_dim
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.skip_connection = skip_connection
        self.dynamic_graph = dynamic_graph
        self.encoder = NRIEncoder(
            node_features=self.node_features, 
            edge_features=self.edge_features, 
            hidden_size=self.hidden_size, 
            skip_connection=self.skip_connection,
            node_embedding_dim=self.node_embedding_dim,
            edge_embedding_dim=self.edge_embedding_dim,
        )
        self.decoder = NRIDecoder(
            seq_len=self.seq_len,
            node_features=self.node_features,
            dynamic_graph=self.dynamic_graph,
            encoder=self.encoder,
            edge_embedding_dim=self.edge_embedding_dim,
            hidden_size=self.hidden_size,
        )

    def forward(self, batch):
        edge_embedding = self.encoder(batch)
        z = torch.nn.functional.gumbel_softmax(edge_embedding, tau=0.5)
        output = self.decoder(batch.x, batch.edge_index, z)
        return output, F.softmax(edge_embedding, dim=-1)
    
### VAE MODULES 

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
    
class Decoder(torch.nn.Module):
    """Decoder from graph to predicted positions"""
    def __init__(self, input_size, hidden_size, output_size, seq_len, edge_embedding_dim, sampling, recurrent):
        super(Decoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.seq_len = seq_len
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

### NRI MODULES

class NRIEncoder(torch.nn.Module):
    """Encoder for fully-connected graph inputs"""
    def __init__(self, node_features, edge_features, hidden_size, skip_connection, node_embedding_dim, edge_embedding_dim):
        super(NRIEncoder, self).__init__()
        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_size  = hidden_size
        self.node_embedding_dim = node_embedding_dim
        self.edge_embedding_dim = edge_embedding_dim
        self.skip_connection = skip_connection
        self.node_embedding_eqn_5 = Sequential(Linear(self.node_features, self.node_embedding_dim), ReLU())
        self.mlp_eqn_6 = Sequential(Linear(2*self.node_embedding_dim, self.hidden_size), 
                      ReLU(), 
                      Linear(self.hidden_size, self.node_embedding_dim))
        self.mlp_eqn_7 = Sequential(Linear(self.node_embedding_dim, self.hidden_size), 
                  ReLU(), 
                  Linear(self.hidden_size, self.node_embedding_dim))
        if self.skip_connection:
            self.mlp_eqn_8 = Sequential(Linear(4*self.node_embedding_dim, self.hidden_size), 
                          ReLU(), 
                          Linear(self.hidden_size, self.edge_embedding_dim))
        else:
            self.mlp_eqn_8 = Sequential(Linear(2*self.node_embedding_dim, self.hidden_size), 
                          ReLU(), 
                          Linear(self.hidden_size, self.edge_embedding_dim))
        self.graph_conv = NRIGraphConv(in_channels=self.node_embedding_dim, 
                                        out_channels=self.node_embedding_dim, 
                                        nn=self.mlp_eqn_6, nn_2=self.mlp_eqn_7, 
                                        root_weight=False, 
                                        bias=False, 
                                        aggr='add')

    def forward(self, data):
        node_embedding = self.node_embedding_eqn_5(data.x)
        
        if self.skip_connection:
            x_skip = node_embedding
            source_node_features_skip = torch.index_select(x_skip, 0, data.edge_index[0])
            destination_node_features_skip = torch.index_select(x_skip, 0, data.edge_index[1])
            edge_messages_skip = torch.cat([source_node_features_skip, destination_node_features_skip], dim=1)
        
        node_embedding = self.graph_conv(node_embedding, data.edge_index)
        source_node_features = torch.index_select(node_embedding, 0, data.edge_index[0])
        destination_node_features = torch.index_select(node_embedding, 0, data.edge_index[1])
        edge_messages = torch.cat([source_node_features, destination_node_features], dim=1)
        
        if self.skip_connection:
            final_concat = torch.cat([edge_messages, edge_messages_skip], axis=1)
            edge_embedding = self.mlp_eqn_8(final_concat)
        else:
            edge_embedding = self.mlp_eqn_8(edge_messages)
        return edge_embedding
    
    
class NRIGraphConv(MessagePassing):
    """Heavily inspired by NNConv; used for the NRI Encoder."""

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
        edge_features = torch.cat([x_i,x_j], dim=1).detach().clone()
        edge_features = self.nn(edge_features)
        return edge_features

    def update(self, aggr_out, x):
        aggr_out = self.nn_2(aggr_out) 
        
        if self.root is not None:
            aggr_out = aggr_out + torch.mm(x, self.root)
        if self.bias is not None:
            aggr_out = aggr_out + self.bias

        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels, self.out_channels)


class NRIDecoder(torch.nn.Module):
    """Decoder from graph to predicted positions"""
    def __init__(self, node_features, hidden_size, seq_len, dynamic_graph, encoder, edge_embedding_dim):
        super(NRIDecoder, self).__init__()
        self.node_features = node_features
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.edge_embedding_dim = edge_embedding_dim
        self.dynamic_graph = dynamic_graph
        self.encoder = encoder
        self.rnn_graph_conv = NRIDecoder_Recurrent(node_features=self.node_features, 
                                                   seq_len=self.seq_len, 
                                                   dynamic_graph=self.dynamic_graph,
                                                   encoder=self.encoder,
                                                   k=self.edge_embedding_dim, 
                                                   hidden_size=self.hidden_size,
                                                   f_out=Sequential(Linear(self.hidden_size, self.node_features), ReLU()),
                                                   f_out_2=Sequential(Linear(self.hidden_size, self.node_features), ReLU()),
                                                  )
    def forward(self, x, edge_index, z):
        h = self.rnn_graph_conv(x, edge_index, z)
        mus = x + h
        return mus
    
class NRIDecoder_Recurrent(MessagePassing):
    """Adapted from GatedGraphConv layer."""
    def __init__(self, node_features: int, seq_len: int, k: int, f_out, f_out_2, hidden_size: int, encoder: None, dynamic_graph: bool = False, aggr: str = 'add', bias: bool = True, **kwargs):
        super(NRIDecoder_Recurrent, self).__init__(aggr=aggr, **kwargs)
        self.node_features = node_features
        self.seq_len = seq_len
        self.k = k
        self.rnn = torch.nn.GRUCell(node_features, hidden_size, bias=bias)
        self.f_out = f_out
        self.f_out_2 = f_out_2
        self.dynamic_graph = dynamic_graph
        self.encoder = encoder
        self.mlp_list = [Sequential(Linear(2*node_features, hidden_size), ReLU(), Linear(hidden_size, hidden_size)) for i in range(k)]
        self.reset_parameters()

    def reset_parameters(self):
        self.rnn.reset_parameters()

    def forward(self, x, edge_index, z):
        if x.size(-1) > self.node_features:
            raise ValueError('The number of input channels is not allowed to be larger than the number of output channels')
        
        if x.size(-1) < self.node_features:
            zero = x.new_zeros(x.size(0), self.node_features - x.size(-1))
            x = torch.cat([x, zero], dim=1)
                    
        h = torch.zeros(x.size())
        for timestep in range(self.seq_len):
            if self.dynamic_graph: 
                ### Resample to get a new z at each timestep 
                batch = Data(x=x, edge_index=edge_index)
                edge_embedding = self.encoder(batch)
                z = torch.nn.functional.gumbel_softmax(edge_embedding, tau=0.5)
            if h.size() == x.size():
                m = self.propagate(x=h, edge_index=edge_index, z=z, size=None)
            else:
                m = self.propagate(x=self.f_out(h), edge_index=edge_index, z=z, size=None)
            h = self.rnn(x, m)
            x = self.f_out_2(h)
        return x

    def message(self, x_i, x_j, z):
        edge_features = torch.cat([x_i,x_j], dim=1).detach().clone()
        k_list = [z[:,i].view(-1, 1)*layer(edge_features) for (i, layer) in enumerate(self.mlp_list)] # element-wise multiplication
        stack = torch.stack(k_list, dim=1)
        output = stack.sum(dim=1) # sum over k
        return output

    def __repr__(self):
        return '{}({}, seq_len={})'.format(self.__class__.__name__,
                                              self.node_features,
                                              self.seq_len)
    
    
