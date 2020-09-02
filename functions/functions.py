import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
import math
import pickle
import json
import time

### LOSS FUNCTIONS ########################################################

def gaussian_neg_log_likelihood(x, mu, sigma):
    gaussian = MultivariateNormal(x - mu, 2*sigma**2*torch.eye(x.size(1), device=x.device))
    return gaussian.log_prob(gaussian.sample()).sum() / (x.size(0) * x.size(1))

# def gaussian_neg_log_likelihood(x, mu, sigma):
#     print(x.shape)
#     print(mu.shape)
#     sum_gaussian = torch.sum((x - mu)**2, axis=0)/2*sigma**2
#     constant = (len(sum_gaussian)/2)*np.log(2*np.pi*sigma**2)
#     return sum_gaussian + constant

def nll_gaussian(preds, target, variance=5e-5, add_const=False):
    neg_log_p = ((preds - target) ** 2 / (2 * variance))
#     print("preds - target: Min:{:.6f} Max:{:.6f}".format((preds-target).min().item(), (preds-target).max().item()))
#     print("numerator:", ((preds - target) ** 2).sum().item())
#     print("neg_log_p sum:",neg_log_p.sum().item())
    if add_const:
        const = 0.5 * np.log(2 * np.pi * variance) # shouldn't this be multiplied by n?
        neg_log_p += const
    return neg_log_p.sum() / (target.size(0) * target.size(1))

def kl_categorical(preds, log_prior, num_atoms, eps=1e-16):
    kl_div = preds * (torch.log(preds + eps) - log_prior)
    return kl_div.sum() / (num_atoms * preds.size(0))

def kl_categorical_uniform(preds, num_atoms, num_edge_types, add_const=False, eps=1e-16):
    kl_div = preds * torch.log(preds + eps) # Shannon entropy -- n log n
    if add_const:
        const = np.log(num_edge_types)
        kl_div += const
    return kl_div.sum() / (num_atoms * preds.size(0))

### MODEL SUMMARIES #######################################################

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
            
def gnn_model_summary(model):
    model_params_list = list(model.named_parameters())
    print("----------------------------------------------------------------")
    line_new = "{:>20}  {:>25} {:>15}".format("Layer.Parameter", "Param Tensor Shape", "Param #")
    print(line_new)
    print("----------------------------------------------------------------")
    for elem in model_params_list:
        p_name = elem[0] 
        p_shape = list(elem[1].size())
        p_count = torch.tensor(elem[1].size()).prod().item()
        line_new = "{:>20}  {:>25} {:>15}".format(p_name, str(p_shape), str(p_count))
        print(line_new)
    print("----------------------------------------------------------------")
    total_params = sum([param.nelement() for param in model.parameters()])
    print("Total params: {:,}".format(total_params))
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Trainable params: {:,}".format(num_trainable_params))
    print("Non-trainable params: {:,}".format(total_params - num_trainable_params))
    
### RESETTING ROOT_WEIGHT & BIAS ###########################################

def uniform(size, tensor):
    bound = 1.0 / math.sqrt(size)
    if tensor is not None:
        tensor.data.uniform_(-bound, bound)
        
def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)
        
def reset(nn):
    def _reset(item):
        if hasattr(item, 'reset_parameters'):
            item.reset_parameters()

    if nn is not None:
        if hasattr(nn, 'children') and len(list(nn.children())) > 0:
            for item in nn.children():
                _reset(item)
        else:
            _reset(nn)
