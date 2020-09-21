import torch
from torch.utils.data import Dataset, SequentialSampler
from torch.utils.data.dataset import TensorDataset
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.utils.convert import to_networkx
torch.manual_seed(0)

import networkx as nx # for visualizing graphs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import pdb
import os 
import argparse
import pickle
import json
import time

from functions.load_data import MarielDataset, edges
from functions.functions import *
from functions.modules import *

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default="vae", help='Distinguishing prefix for save directory.')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--batch_size', type=int, default=128, help='Number of sequences per batch.')
parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate for training.')
parser.add_argument('--seq_len', type=int, default=49, help='Number of timesteps per sequence.')
parser.add_argument('--node_embedding_dim', type=int, default=256, help='Node embedding size.')
parser.add_argument('--edge_embedding_dim', type=int, default=4, help='Edge embedding size (i.e. number of edge types).')
parser.add_argument('--hidden_size', type=int, default=256, help='Number of timesteps per sequence.')
parser.add_argument('--num_layers', type=int, default=1, help='Number of recurrent layers in decoder.')
parser.add_argument('--pred_to_reco_ratio', type=float, default=1., help='How to weight prediction versus reconstruction losses during training.')
parser.add_argument('--predicted_timesteps', type=int, default=10, help='Number of timesteps to predict. Must be < seq_len.')
parser.add_argument('--batch_limit', type=int, default=0, help='Number of batches per epoch -- if 0, will run over all batches.')
parser.add_argument('--reduced_joints', action='store_true', default=False, help='Trains on 18 joints rather than all 53.')
parser.add_argument('--skip_connection', action='store_true', default=False, help='Enables skip connection in the encoder.')
parser.add_argument('--dynamic_graph',action='store_true', default=False, help='Enables dynamic graph re-computation per timestep during testing.')
parser.add_argument('--no_overlap', action='store_true', default=False, help="Don't train on overlapping sequences.")
parser.add_argument('--no_cuda', action='store_true', default=False, help="Don't use GPU, even if available.")
parser.add_argument('--sparsity_prior', action='store_true', default=False, help="Enables sparsity prior when training.")
parser.add_argument('--shuffle', action='store_true', default=False, help="Enables shuffling samples in the DataLoader.")
args = parser.parse_args()
print(args)

if args.reduced_joints: 
    n_joints = 18
else:
    n_joints = 53

save_folder = os.path.join("logs",args.name)
if not os.path.exists(save_folder): 
    os.makedirs(save_folder)
checkpoint_path = os.path.join(save_folder,"best_weights.pth")
log_file = os.path.join(save_folder, 'log.txt')
log = open(log_file, 'w')
print(args, file=log)
print("Save folder: {}".format(save_folder), file=log)
log.flush()

### LOAD DATA
data = MarielDataset(seq_len=args.seq_len, reduced_joints=args.reduced_joints, predicted_timesteps=args.predicted_timesteps, no_overlap=args.no_overlap)

train_indices = np.arange(int(0.7*len(data))) # 70% split for training data, no shuffle
val_indices = np.arange(int(0.7*len(data)),int(0.85*len(data))) # next 15% on validation
test_indices = np.arange(int(0.85*len(data)), len(data)) # last 15% on test

train = torch.utils.data.Subset(data, train_indices)
val = torch.utils.data.Subset(data, val_indices)
test = torch.utils.data.Subset(data, test_indices)

dataloader_train = DataLoader(train, batch_size=args.batch_size, shuffle=args.shuffle, drop_last=True)
dataloader_val = DataLoader(val, batch_size=args.batch_size, shuffle=args.shuffle, drop_last=True)
dataloader_test = DataLoader(test, batch_size=args.batch_size, shuffle=args.shuffle, drop_last=True)

pickle.dump({'args': args}, open(os.path.join(save_folder,'args.pkl'), "wb"))
torch.save(dataloader_train, os.path.join(save_folder, 'dataloader_train.pth'))
torch.save(dataloader_val, os.path.join(save_folder, 'dataloader_val.pth'))
torch.save(dataloader_test, os.path.join(save_folder, 'dataloader_test.pth'))

print("\nGenerated {:,} training batches of shape: {}".format(len(dataloader_train), data[0]), file=log)
log.flush()
print("\nGenerated {:,} training batches of shape: {}".format(len(dataloader_train), data[0]))

if args.no_cuda:
    device = 'cpu'
else:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
print("\nUsing {}".format(device))
print("\nUsing {}".format(device), file=log)

### DEFINE MODEL 
node_features = data.seq_len*data.n_dim
edge_features = data[0].num_edge_features
checkpoint_loaded = False 

encoder = NRIEncoder(
            node_features=node_features, 
            edge_features=edge_features, 
            hidden_size=args.hidden_size, 
            skip_connection=args.skip_connection,
            node_embedding_dim=args.node_embedding_dim,
            edge_embedding_dim=args.edge_embedding_dim,
        )


model = NRI(device=device,
            node_features=node_features, 
            edge_features=edge_features, 
            hidden_size=args.hidden_size, 
            node_embedding_dim=args.node_embedding_dim,
            edge_embedding_dim=args.edge_embedding_dim,
            seq_len=args.seq_len,
            predicted_timesteps=args.predicted_timesteps,
            skip_connection=args.skip_connection,
            dynamic_graph=args.dynamic_graph,
           )

optimizer = torch.optim.Adam(list(model.parameters()), lr=args.lr, weight_decay=5e-4)

model = model.to(device)
print(model, file=log)
print(model)
print("Encoder trainable parameters: {:,}".format(count_parameters(encoder)))
print("Encoder trainable parameters: {:,}".format(count_parameters(encoder)), file=log)
print("Total trainable parameters: {:,}".format(count_parameters(model)))
print("Total trainable parameters: {:,}".format(count_parameters(model)), file=log)
log.flush()

if args.sparsity_prior:
    prior_array = []
    prior_array.append(0.9) # 90% of edges as non-edge
    for k in range(1,args.edge_embedding_dim):
        prior_array.append(0.1/(args.edge_embedding_dim-1))
    prior = np.array(prior_array)
    print("Using sparsity prior: {}".format(prior))
    log_prior = torch.FloatTensor(np.log(prior))
    if torch.cuda.is_available() and device != 'cpu': log_prior = log_prior.cuda()

### LOAD PRE-TRAINED WEIGHTS
if os.path.isfile(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss_checkpoint = checkpoint['loss']
    checkpoint_loaded = True
    print("Loading saved checkpoint from {} (best loss so far: {:.5f})...".format(checkpoint_path, loss_checkpoint), file=log)
    log.flush()
    print("Loading saved checkpoint from {} (best loss so far: {:.5f})...".format(checkpoint_path, loss_checkpoint))

### TRAIN
mse_loss = torch.nn.MSELoss(reduction='mean')
prediction_to_reconstruction_loss_ratio = args.pred_to_reco_ratio # you might want to weight the prediction loss higher to help it compete with the larger prediction seq_len

def train_model(epochs):
    train_losses = []
    train_mse_losses = []
    train_nll_losses = []
    train_kl_losses = []
    val_losses = []
    val_mse_losses = []
    val_nll_losses = []
    val_kl_losses = []
    
    for epoch in range(epochs):
        model.train()
        t = time.time()
        n_batches = 0
        n_val_batches = 0
        total_train_loss = 0
        total_train_mse_loss = 0
        total_train_nll_loss = 0
        total_train_kl_loss = 0
        total_val_loss = 0
        total_val_mse_loss = 0
        total_val_nll_loss = 0
        total_val_kl_loss = 0
        
        ### TRAINING LOOP
        for batch in tqdm(dataloader_train, desc="Train"):
            batch = batch.to(device)
            print("Batch is size {} -- first 10 timesteps for joint 0_x: {}".format(batch.x.size(), batch.x[0,:10]))
            
            ### CALCULATE MODEL OUTPUTS
            output, edge_types, logits, probabilities = model(batch)

            ### CALCULATE LOSS
            train_mse_loss = mse_loss(output, batch.x.to(device)) # just calculate this for comparison; it's not added to the loss
            train_nll_loss = nll_gaussian(output, batch.x.to(device))
            if args.sparsity_prior:
                train_kl_loss = kl_categorical(probabilities, log_prior, 53)
            else:
                train_kl_loss = kl_categorical_uniform(probabilities, 53, args.edge_embedding_dim)
            train_loss = train_nll_loss + train_kl_loss

            ### ADD LOSSES TO TOTALS
            total_train_loss += train_loss.item()
            total_train_mse_loss += train_mse_loss.item()
            total_train_nll_loss += train_nll_loss.item()
            total_train_kl_loss += train_kl_loss.item()
            
            ### BACKPROPAGATE
            optimizer.zero_grad() # reset the gradients to zero
            train_loss.backward()
            optimizer.step()

            ### OPTIONAL -- STOP TRAINING EARLY
            n_batches += 1
            if (args.batch_limit > 0) and (n_batches >= args.batch_limit): break # for shorter iterations during testing
        
        ### VALIDATION LOOP
        model.eval()
        for batch in tqdm(dataloader_val, desc="Val"):
            batch = batch.to(device)
            
            ### CALCULATE MODEL OUTPUTS
            output, edge_types, logits, probabilities = model(batch)
            
            ### CALCULATE LOSS
            val_mse_loss = mse_loss(output, batch.x.to(device)) # just for comparison
            val_nll_loss = nll_gaussian(output, batch.x.to(device))
            if args.sparsity_prior:
                val_kl_loss = kl_categorical(probabilities, log_prior, 53)
            else:
                val_kl_loss = kl_categorical_uniform(probabilities, 53, args.edge_embedding_dim)
            val_kl_loss = kl_categorical_uniform(probabilities, 53, args.edge_embedding_dim)
            val_loss = val_nll_loss # note: don't add KL loss

            ### ADD LOSSES TO TOTALS
            total_val_loss += val_loss.item()
            total_val_mse_loss += val_mse_loss.item()
            total_val_nll_loss += val_nll_loss.item()
            total_val_kl_loss += val_kl_loss.item()

            ### OPTIONAL -- STOP TRAINING EARLY
            n_val_batches += 1
            if (args.batch_limit > 0) and (n_val_batches >= args.batch_limit): break # temporary -- for stopping training early
        
        ### CALCULATE AVERAGE LOSSES PER EPOCH   
        epoch_train_loss = total_train_loss / n_batches
        epoch_train_mse_loss = total_train_mse_loss / n_batches
        epoch_train_nll_loss = total_train_nll_loss / n_batches
        epoch_train_kl_loss = total_train_kl_loss / n_batches

        train_losses.append(epoch_train_loss) 
        train_mse_losses.append(epoch_train_mse_loss)
        train_nll_losses.append(epoch_train_nll_loss)
        train_kl_losses.append(epoch_train_kl_loss)

        epoch_val_loss = total_val_loss / n_batches
        epoch_val_mse_loss = total_val_mse_loss / n_batches
        epoch_val_nll_loss = total_val_nll_loss / n_batches
        epoch_val_kl_loss = total_val_kl_loss / n_batches

        val_losses.append(epoch_val_loss) 
        val_mse_losses.append(epoch_val_mse_loss)
        val_nll_losses.append(epoch_val_nll_loss)
        val_kl_losses.append(epoch_val_kl_loss)

        ### Print to log file
        print("epoch : {}/{} | train_loss = {:,.5f} | train_mse_loss: {:,.5f} | train_nll_loss: {:,.5f} | train_kl_loss = {:,.5f} | val_loss = {:,.5f} | val_mse_loss: {:,.5f} | val_nll_loss: {:,.5f} | val_kl_loss: {:,.5f} | time: {:.1f} sec".format(
            epoch+1, 
            epochs, 
            epoch_train_loss,
            epoch_train_mse_loss,
            epoch_train_nll_loss,
            epoch_train_kl_loss,
            epoch_val_loss,
            epoch_val_mse_loss,
            epoch_val_nll_loss,
            epoch_val_kl_loss,
            time.time() - t),
            file=log)
        log.flush()
        ### Print to console
        print("epoch : {}/{} | train_loss = {:,.5f} | train_mse_loss: {:,.5f} | train_nll_loss: {:,.5f} | train_kl_loss = {:,.5f} | val_loss = {:,.5f} | val_mse_loss: {:,.5f} | val_nll_loss: {:,.5f} | val_kl_loss: {:,.5f} | time: {:.1f} sec".format(
            epoch+1, 
            epochs, 
            epoch_train_loss,
            epoch_train_mse_loss,
            epoch_train_nll_loss,
            epoch_train_kl_loss,
            epoch_val_loss,
            epoch_val_mse_loss,
            epoch_val_nll_loss,
            epoch_val_kl_loss,
            time.time() - t),
            )
        
        if epoch == 0:
            ### Checkpoint the first epoch
            if checkpoint_loaded: 
                best_loss = min(epoch_val_loss, loss_checkpoint)
            else:
                best_loss = epoch_val_loss
                torch.save({
                 'epoch': epoch,
                 'model_state_dict': model.state_dict(),
                 'optimizer_state_dict': optimizer.state_dict(),
                 'loss': best_loss,
                 }, checkpoint_path)
                print("Saved model checkpoint to {}.".format(checkpoint_path), file=log)
                log.flush()
                print("Saved model checkpoint to {}.".format(checkpoint_path))
                        
        if epoch_val_loss < best_loss:
            best_loss = epoch_val_loss
            torch.save({
             'epoch': epoch,
             'model_state_dict': model.state_dict(),
             'optimizer_state_dict': optimizer.state_dict(),
             'loss': best_loss,
             }, checkpoint_path)
            print("Better loss achieved -- saved model checkpoint to {}.".format(checkpoint_path), file=log)
            log.flush()
            print("Better loss achieved -- saved model checkpoint to {}.".format(checkpoint_path))

    loss_dict = {
    "train_losses": train_losses,
    "train_mse_losses": train_mse_losses,
    "train_nll_losses": train_nll_losses,
    "train_kl_losses": train_kl_losses,
    "val_losses": val_losses,
    "val_mse_losses": val_mse_losses,
    "val_nll_losses": val_nll_losses,
    "val_kl_losses": val_kl_losses,
    }

    with open(os.path.join(save_folder,'losses.json'), 'w') as f:
	    json.dump(loss_dict, f)

train_model(epochs=args.epochs)


