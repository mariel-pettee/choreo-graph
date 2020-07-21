import torch
from torch.utils.data import Dataset, SequentialSampler
from torch.utils.data.dataset import TensorDataset
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.utils.convert import to_networkx

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
parser.add_argument('--batch_size', type=int, default=32, help='Number of sequences per batch.')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for training.')
parser.add_argument('--seq_len', type=int, default=49, help='Number of timesteps per sequence.')
parser.add_argument('--node_embedding_dim', type=int, default=64, help='Node embedding size.')
parser.add_argument('--edge_embedding_dim', type=int, default=32, help='Edge embedding size (i.e. number of edge types).')
parser.add_argument('--hidden_size', type=int, default=64, help='Number of timesteps per sequence.')
parser.add_argument('--num_layers', type=int, default=3, help='Number of recurrent layers in decoder.')
parser.add_argument('--predicted_timesteps', type=int, default=10, help='Number of timesteps to predict.')
parser.add_argument('--batch_limit', type=int, default=0, help='Number of batches per epoch -- if 0, will run over all batches.')
parser.add_argument('--reduced_joints', action='store_true', default=False, help='Trains on 18 joints rather than all 53.')
parser.add_argument('--no_overlap', action='store_true', default=False, help="Don't train on overlapping sequences.")
parser.add_argument('--sampling', action='store_true', default=False, help="Enables sampling step between encoder & decoder.")
parser.add_argument('--recurrent', action='store_true', default=False, help="Enables recurrent decoder.")
args = parser.parse_args()
print(args)

if args.reduced_joints: 
    n_joints = 18
else:
    n_joints = 53

save_folder = os.path.join("logs",args.name+"_{}joints_seqlen{}_pred{}".format(n_joints, args.seq_len, args.predicted_timesteps))
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

dataloader_train = DataLoader(data, batch_size=args.batch_size, shuffle=False, drop_last=True, sampler=SequentialSampler(train_indices))
dataloader_val = DataLoader(data, batch_size=args.batch_size, shuffle=False, drop_last=True, sampler=SequentialSampler(val_indices))
dataloader_test = DataLoader(data, batch_size=args.batch_size, shuffle=False, drop_last=True, sampler=SequentialSampler(test_indices))

torch.save(dataloader_train, os.path.join(save_folder, 'dataloader_train.pth'))
torch.save(dataloader_val, os.path.join(save_folder, 'dataloader_val.pth'))
torch.save(dataloader_test, os.path.join(save_folder, 'dataloader_test.pth'))

print("\nGenerated {:,} training batches of shape: {}".format(len(dataloader_train), data[0]))

### DEFINE MODEL 
node_features = data.seq_len*data.n_dim
edge_features = data[0].num_edge_features
node_embedding_dim = args.node_embedding_dim
edge_embedding_dim = args.edge_embedding_dim # number of edge types
hidden_size = args.hidden_size
num_layers = args.num_layers
checkpoint_loaded = False 

model = VAE(node_features=node_features, 
            edge_features=edge_features, 
            hidden_size=hidden_size, 
            node_embedding_dim=node_embedding_dim,
            edge_embedding_dim=edge_embedding_dim,
            num_layers=num_layers,
            input_size=node_embedding_dim, 
            output_size=node_features+args.predicted_timesteps*data.n_dim,
            sampling=args.sampling, 
            recurrent=args.recurrent,
           )

optimizer = torch.optim.Adam(list(model.parameters()), lr=args.lr, weight_decay=5e-4)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("\nUsing {}".format(device), file=log)
model = model.to(device)
print(model, file=log)
print("Total trainable parameters: {:,}".format(count_parameters(model)), file=log)
log.flush()

### LOAD PRE-TRAINED WEIGHTS
if os.path.isfile(checkpoint_path):
    print("Loading saved checkpoint from {}...".format(checkpoint_path), file=log)
    log.flush()
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss_checkpoint = checkpoint['loss']
    checkpoint_loaded = True

### TRAIN
mse_loss = torch.nn.MSELoss(reduction='mean')
prediction_to_reconstruction_loss_ratio = 0 # you might want to weight the prediction loss higher to help it compete with the larger prediction seq_len

def train_model(epochs):
    train_losses = []
    train_reco_losses = []
    train_pred_losses = []
    val_losses = []
    val_reco_losses = []
    val_pred_losses = []
    for epoch in range(epochs):
        model.train()
        t = time.time()
        n_batches = 0
        total_train_loss = 0
        total_train_reco_loss = 0
        total_train_pred_loss = 0
        total_val_loss = 0
        total_val_reco_loss = 0
        total_val_pred_loss = 0
        
        ### TRAINING LOOP
        for batch in dataloader_train:
            batch = batch.to(device)
            
            ### CALCULATE MODEL OUTPUTS
            output = model(batch)
            
            ### CALCULATE LOSS
            train_reco_loss = mse_loss(batch.x.to(device), output[:,:node_features]) # compare first seq_len timesteps.item()
            if args.predicted_timesteps > 0: 
                train_pred_loss = prediction_to_reconstruction_loss_ratio*mse_loss(batch.y.to(device), output[:,node_features:]) # compare last part to unseen data
                train_loss = train_reco_loss + train_pred_loss
            else:
                train_loss = train_reco_loss

            ### ADD LOSSES TO TOTALS
            total_train_loss += train_loss.item()
            total_train_reco_loss += train_reco_loss.item()
            if args.predicted_timesteps > 0: 
                total_train_pred_loss += train_pred_loss.item()

            ### BACKPROPAGATE
            optimizer.zero_grad() # reset the gradients to zero
            train_loss.backward()
            optimizer.step()

            ### OPTIONAL -- STOP TRAINING EARLY
            n_batches += 1
            if (args.batch_limit > 0) and (n_batches >= args.batch_limit): break # temporary -- for stopping training early
        
        ### VALIDATION LOOP
        model.eval()
        for batch in dataloader_val:
            batch = batch.to(device)
            
            ### CALCULATE MODEL OUTPUTS
            output = model(batch)
            
            ### CALCULATE LOSS
            val_reco_loss = mse_loss(batch.x.to(device), output[:,:node_features]) # compare first seq_len timesteps.item()
            if args.predicted_timesteps > 0: 
                val_pred_loss = prediction_to_reconstruction_loss_ratio*mse_loss(batch.y.to(device), output[:,node_features:]) # compare last part to unseen data
                val_loss = val_reco_loss + val_pred_loss
            else:
                val_loss = val_reco_loss

            ### ADD LOSSES TO TOTALS
            total_val_loss += val_loss.item()
            total_val_reco_loss += val_reco_loss.item()
            if args.predicted_timesteps > 0: 
                total_val_pred_loss += val_pred_loss.item()

            ### OPTIONAL -- STOP TRAINING EARLY
            n_batches += 1
            if (args.batch_limit > 0) and (n_batches >= args.batch_limit): break # temporary -- for stopping training early
        
        ### CALCULATE AVERAGE LOSSES PER EPOCH   
        epoch_train_loss = total_train_loss / n_batches
        epoch_train_reco_loss = total_train_reco_loss / n_batches
        epoch_train_pred_loss = total_train_pred_loss / n_batches

        train_losses.append(epoch_train_loss) 
        train_reco_losses.append(epoch_train_reco_loss)
        train_pred_losses.append(epoch_train_pred_loss)

        epoch_val_loss = total_val_loss / n_batches
        epoch_val_reco_loss = total_val_reco_loss / n_batches
        epoch_val_pred_loss = total_val_pred_loss / n_batches

        val_losses.append(epoch_val_loss) 
        val_reco_losses.append(epoch_val_reco_loss)
        val_pred_losses.append(epoch_val_pred_loss)

        print("epoch : {}/{} | train_loss = {:,.4f} | train_reco_loss: {:,.4f} | train_pred_loss: {:,.4f} | val_loss = {:,.4f} | val_reco_loss: {:,.4f} | val_pred_loss: {:,.4f} |time: {:.4f} sec".format(epoch+1, epochs, 
                                                                                                                epoch_train_loss,
                                                                                                                epoch_train_reco_loss, 
                                                                                                                epoch_train_pred_loss,
                                                                                                                epoch_val_loss,
                                                                                                                epoch_val_reco_loss, 
                                                                                                                epoch_val_pred_loss,
                                                                                                                time.time() - t),
                                                                                                                file=log)
        log.flush()
        
        if epoch == 0 and not checkpoint_loaded: best_loss = epoch_val_loss
        elif epoch == 0 and checkpoint_loaded: best_loss = min(epoch_val_loss, loss_checkpoint)
            
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

    loss_dict = {
	"train_losses": train_losses,
	"train_reco_losses": train_reco_losses,
	"train_pred_losses": train_pred_losses,
	"val_losses": val_losses,
	"val_reco_losses": val_reco_losses,
	"val_pred_losses": val_pred_losses,
			}

    with open(os.path.join(save_folder,'losses.json'), 'w') as f:
	    json.dump(loss_dict, f)

train_model(epochs=args.epochs)


