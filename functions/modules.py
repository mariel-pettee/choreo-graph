import numpy as np
import torch
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import MessagePassing, GatedGraphConv
from torch_geometric.utils import add_self_loops, degree
from torch.nn import Parameter, ModuleList, BatchNorm1d
from torch.autograd import Variable
import torch.nn.functional as F
from .functions import *
from torch_geometric.data import Data
# import pdb; pdb.set_trace()
import time

class NRI(torch.nn.Module):
    """Implementation of NRI with Pytorch Geometric"""
    def __init__(self, device, node_features, edge_features, hidden_size, skip_connection, node_embedding_dim, edge_embedding_dim, dynamic_graph, seq_len, predicted_timesteps):
        super(NRI, self).__init__()
        self.device = device
        self.node_features = node_features
        self.node_embedding_dim = node_embedding_dim
        self.edge_features = edge_features
        self.edge_embedding_dim = edge_embedding_dim
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.predicted_timesteps = predicted_timesteps
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
            device=self.device,
            seq_len=self.seq_len,
            predicted_timesteps=self.predicted_timesteps,
            node_features=self.node_features,
            dynamic_graph=self.dynamic_graph,
            encoder=self.encoder,
            edge_embedding_dim=self.edge_embedding_dim,
            hidden_size=self.hidden_size,
        )

    def forward(self, batch):
        edge_embedding = self.encoder(batch)
        z = torch.nn.functional.gumbel_softmax(edge_embedding, tau=0.5, hard=True)
        output = self.decoder(batch.x, batch.edge_index, z)
        return output, z, edge_embedding, F.softmax(edge_embedding, dim=-1)
    
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
        self.node_embedding_eqn_5 = Sequential(Linear(self.node_features, self.node_embedding_dim), ReLU(), Linear(self.node_embedding_dim, self.node_embedding_dim), ReLU(), BatchNorm1d(self.node_embedding_dim, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        self.mlp_eqn_6 = Sequential(Linear(2*self.node_embedding_dim, self.hidden_size), 
                      ReLU(), 
                      Linear(self.hidden_size, self.node_embedding_dim),
                      ReLU(), 
                      BatchNorm1d(self.node_embedding_dim, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        self.mlp_eqn_7 = Sequential(Linear(self.node_embedding_dim, self.hidden_size), 
                  ReLU(), 
                  Linear(self.hidden_size, self.node_embedding_dim),
                  ReLU(), 
                  BatchNorm1d(self.node_embedding_dim, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        if self.skip_connection:
            self.mlp_eqn_8 = Sequential(Linear(4*self.node_embedding_dim, self.hidden_size), 
                          ReLU(), 
                          Linear(self.hidden_size, self.hidden_size),
                          ReLU(),
                          BatchNorm1d(self.hidden_size, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                          Linear(self.hidden_size, self.edge_embedding_dim),
                                       )
        else:
            self.mlp_eqn_8 = Sequential(Linear(2*self.node_embedding_dim, self.hidden_size), 
                          ReLU(), 
                          Linear(self.hidden_size, self.hidden_size),
                          ReLU(),
                          BatchNorm1d(self.hidden_size, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                          Linear(self.hidden_size, self.edge_embedding_dim),
                                       )
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
    def __init__(self, device, node_features, hidden_size, seq_len, predicted_timesteps, dynamic_graph, encoder, edge_embedding_dim):
        super(NRIDecoder, self).__init__()
        self.device = device
        self.node_features = node_features
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.predicted_timesteps = predicted_timesteps
        self.edge_embedding_dim = edge_embedding_dim
        self.dynamic_graph = dynamic_graph
        self.encoder = encoder
        self.f_out = Sequential(Linear(self.hidden_size, int(self.node_features/self.seq_len)), ReLU())
        self.rnn_graph_conv = NRIDecoder_Recurrent(device=self.device,
                                                   node_features=self.node_features, 
                                                   seq_len=self.seq_len, 
                                                   dynamic_graph=self.dynamic_graph,
                                                   encoder=self.encoder,
                                                   k=self.edge_embedding_dim, 
                                                   hidden_size=self.hidden_size,
                                                   f_out=Sequential(Linear(self.hidden_size, self.node_features), ReLU()),
                                                  )
    def forward(self, x, edge_index, z):
        h = torch.zeros(x.size(0),self.hidden_size) # initialize hidden state
        predictions = []
        
        ### Reshape x to iterate over timesteps more explicitly
        # previous size of x: [batch_size * n_joints, seq_len*6]
        # new size of x: [batch_size * n_joints, seq_len, 6] (6 is from x,y,z and v_x, v_y, v_z)
        x_flat_shape = x.size()
        n_timesteps = int(x.size(1)/6)
        x = torch.reshape(x, [x.size(0),n_timesteps,6])
        
        ### Loop over timesteps of x, redefining h each time. Note that predicted_timesteps must be < seq_len.
        for timestep in range(n_timesteps):
            if timestep < (self.seq_len - self.predicted_timesteps):
                inputs = x[:,timestep,:] # feed in real data up until transition to prediction-only
            else:
                inputs = predictions[timestep-1] # feed in previous prediction
            h = self.rnn_graph_conv(inputs, edge_index, z, h)
            
            ### Final MLP to convert hidden dimension back into node_features
            mu = inputs + self.f_out(h)
#             print("Average hidden:", torch.mean(h).item())
#             print("Average f_out(h):", torch.mean(self.f_out(h)).item())
            predictions.append(mu)

        mus = torch.stack(predictions, dim=1)
        mus = torch.reshape(mus, x_flat_shape)
        return mus
    
class NRIDecoder_Recurrent(MessagePassing):
    """Adapted from GatedGraphConv layer."""
    def __init__(self, device, node_features: int, seq_len: int, k: int, f_out, hidden_size: int, encoder: None, dynamic_graph: bool = False, aggr: str = 'mean', bias: bool = True, **kwargs):
        super(NRIDecoder_Recurrent, self).__init__(aggr=aggr, **kwargs)
        self.device = device
        self.node_features = node_features
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.k = k
        self.rnn = torch.nn.GRUCell(6, hidden_size, bias=bias)
        self.f_out = f_out
        self.dynamic_graph = dynamic_graph
        self.encoder = encoder
        self.mlp_list = [Sequential(Linear(2*node_features, hidden_size), ReLU(), Linear(hidden_size, hidden_size)) for i in range(k-1)] # leave out one for the non-edge
        self.reset_parameters()

    def reset_parameters(self):
        self.rnn.reset_parameters()

    def forward(self, x, edge_index, z, h):
        if self.dynamic_graph: 
            ### Resample to get a new z at each timestep 
            batch = Data(x=x, edge_index=edge_index)
            edge_embedding = self.encoder(batch)
            z = torch.nn.functional.gumbel_softmax(edge_embedding, tau=0.5)
        if torch.cuda.is_available() and self.device != 'cpu': h = h.cuda()
        m = self.propagate(x=self.f_out(h), edge_index=edge_index, z=z, size=None)
        h = self.rnn(x, m)
        return h
    
    def message(self, x_i, x_j, z):
        edge_features = torch.cat([x_i,x_j], dim=1).detach().clone()
        if torch.cuda.is_available() and self.device != 'cpu':
            # Note: use i+1 to skip the first edge type
            k_list = [z[:,i+1].view(-1, 1)*layer.cuda()(edge_features) for (i, layer) in enumerate(self.mlp_list)] # element-wise multiplication
        else:
            k_list = [z[:,i+1].view(-1, 1)*layer(edge_features) for (i, layer) in enumerate(self.mlp_list)] # element-wise multiplication
        stack = torch.stack(k_list, dim=1)
        output = stack.sum(dim=1) # sum over k
        return output

    def __repr__(self):
        return '{}({}, seq_len={})'.format(self.__class__.__name__,
                                              self.node_features,
                                              self.seq_len)
    
    
