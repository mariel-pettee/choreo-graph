import numpy as np
import torch
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import MessagePassing, GatedGraphConv
from torch_geometric.utils import add_self_loops, degree
from torch.nn import Parameter, ModuleList

def nll_gaussian(preds, target, variance, add_const=False):
    neg_log_p = ((preds - target) ** 2 / (2 * variance))
    if add_const:
        const = 0.5 * np.log(2 * np.pi * variance)
        neg_log_p += const
    return neg_log_p.sum() / (target.size(0) * target.size(1))

def gumbel_softmax_sample(logits, tau=1, eps=1e-10):
    """
    NOTE: Stolen from https://github.com/pytorch/pytorch/pull/3341/commits/327fcfed4c44c62b208f750058d14d4dc1b9a9d3
    Draw a sample from the Gumbel-Softmax distribution
    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb
    (MIT license)
    """
    gumbel_noise = sample_gumbel(logits.size(), eps=eps)
    if logits.is_cuda:
        gumbel_noise = gumbel_noise.cuda()
    y = logits + Variable(gumbel_noise)
    return my_softmax(y / tau, axis=-1)

def gumbel_softmax(logits, tau=1, hard=False, eps=1e-10):
    """
    NOTE: Stolen from https://github.com/pytorch/pytorch/pull/3341/commits/327fcfed4c44c62b208f750058d14d4dc1b9a9d3
    Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
      logits: [batch_size, n_class] unnormalized log-probs
      tau: non-negative scalar temperature
      hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
      [batch_size, n_class] sample from the Gumbel-Softmax distribution.
      If hard=True, then the returned sample will be one-hot, otherwise it will
      be a probability distribution that sums to 1 across classes
    Constraints:
    - this implementation only works on batch_size x num_features tensor for now
    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
    (MIT license)
    """
    y_soft = gumbel_softmax_sample(logits, tau=tau, eps=eps)
    if hard:
        shape = logits.size()
        _, k = y_soft.data.max(-1)
        # this bit is based on
        # https://discuss.pytorch.org/t/stop-gradients-for-st-gumbel-softmax/530/5
        y_hard = torch.zeros(*shape)
        if y_soft.is_cuda:
            y_hard = y_hard.cuda()
        y_hard = y_hard.zero_().scatter_(-1, k.view(shape[:-1] + (1,)), 1.0)
        # this cool bit of code achieves two things:
        # - makes the output value exactly one-hot (since we add then
        #   subtract y_soft value)
        # - makes the gradient equal to y_soft gradient (since we strip
        #   all other gradients)
        y = Variable(y_hard - y_soft.data) + y_soft
    else:
        y = y_soft
    return y

class MLPEncoder(torch.nn.Module):
    """Encoder for fully-connected graph inputs"""
    def __init__(self, node_features, edge_features, hidden_size, node_embedding_dim, edge_embedding_dim):
        super(MLPEncoder, self).__init__()
        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_size  = hidden_size
        self.node_embedding_dim = node_embedding_dim
        self.edge_embedding_dim = edge_embedding_dim
        self.node_embedding = Sequential(Linear(self.node_features, self.node_embedding_dim), ReLU(),Linear(self.node_embedding_dim, self.node_embedding_dim), ReLU(),Linear(self.node_embedding_dim, self.node_embedding_dim), ReLU(),Linear(self.node_embedding_dim, self.node_embedding_dim), ReLU(),Linear(self.node_embedding_dim, self.node_embedding_dim), ReLU())
        self.edge_embedding = Sequential(Linear(self.edge_features, self.edge_embedding_dim), ReLU())
        self.MLP = Sequential(Linear(self.edge_embedding_dim, self.hidden_size), 
                              ReLU(), 
                              Linear(self.hidden_size, 2*self.node_embedding_dim*self.node_embedding_dim))
        self.graph_conv_1 = MLPGraphConv(in_channels=self.node_embedding_dim, 
                                           out_channels=self.node_embedding_dim, 
                                           nn=self.MLP)
        self.graph_conv_2 = MLPGraphConv(in_channels=self.node_embedding_dim, 
                                           out_channels=self.node_embedding_dim, 
                                           nn=self.MLP)
        
        self.graph_conv_list = ModuleList([self.graph_conv_1, self.graph_conv_2])

    def forward(self, data):
        node_embedding = self.node_embedding(data.x)
        edge_embedding = self.edge_embedding(data.edge_attr)
        for layer in self.graph_conv_list:
            node_embedding = layer(node_embedding, data.edge_index, edge_embedding)
        return node_embedding, data.edge_index, edge_embedding

class RNNDecoder(torch.nn.Module):
    """Decoder from graph to predicted positions"""
    def __init__(self, input_size, hidden_size, output_size, edge_embedding_dim, num_layers):
        super(RNNDecoder, self).__init__()
        self.input_size = input_size
        self.hidden_size  = hidden_size
        self.output_size = output_size
        self.edge_embedding_dim = edge_embedding_dim
        self.num_layers = num_layers
        self.node_transform = Sequential(Linear(self.input_size, self.output_size), ReLU())
#         self.edge_transform = Sequential(Linear(self.edge_embedding_dim, self.edge_features), ReLU())
        self.graph_conv = GatedGraphConv(out_channels=self.output_size, num_layers=self.num_layers)

    def forward(self, x, edge_index, edge_attr, z_soft):
        node_features = self.node_transform(x)
        edge_attr = z_soft*edge_attr
        node_features = self.graph_conv(node_features, edge_index, edge_weight=None)
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
