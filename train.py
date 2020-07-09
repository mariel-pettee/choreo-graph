import torch
from torch.utils.data import Dataset
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
parser.add_argument('--seq_len', type=int, default=49, help='Number of timesteps per sequence.')
parser.add_argument('--predicted_timesteps', type=int, default=10, help='Number of timesteps to predict.')
parser.add_argument('--batch_limit', type=int, default=0, help='Number of batches per epoch -- if 0, will run over all batches.')
parser.add_argument('--reduced_joints', action='store_true', default=False, help='Trains on 18 joints rather than all 53.')
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
data = MarielDataset(seq_len=args.seq_len, reduced_joints=args.reduced_joints, predicted_timesteps=args.predicted_timesteps)
dataloader = DataLoader(data, batch_size=args.batch_size, shuffle=False, drop_last=True)
print("\nGenerated {:,} batches of shape: {}".format(len(dataloader), data[0]), file=log)

### DEFINE MODEL 
node_features = data.seq_len*data.n_dim
edge_features = data[0].num_edge_features
node_embedding_dim = 25
edge_embedding_dim = 4 # number of edge types
hidden_size = 50
num_layers = 2
checkpoint_loaded = False 

model = VAE(node_features=node_features, 
            edge_features=edge_features, 
            hidden_size=hidden_size, 
            node_embedding_dim=node_embedding_dim,
            edge_embedding_dim=edge_embedding_dim,
            num_layers=num_layers,
            input_size=node_embedding_dim, 
            output_size=node_features+args.predicted_timesteps*data.n_dim,
           )

optimizer = torch.optim.Adam(list(model.parameters()), lr=1e-4, weight_decay=5e-4)
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
sigma = 0.001 # how to pick sigma?

def train(epochs):
    t = time.time()
    losses = []
    reconstruction_losses = []
    prediction_losses = []
    inputs = []
    outputs = []
    model.train()
    for epoch in tqdm(range(epochs)):
        average_loss = 0
        average_reconstruction_loss = 0
        average_prediction_loss = 0
        i = 0
        batch_inputs = []
        batch_outputs = []
        
        for batch in tqdm(dataloader, desc="Batches"):
            batch = batch.to(device)
            optimizer.zero_grad() # reset the gradients to zero
            
            ### CALCULATE MODEL OUTPUTS
            output = model(batch)
            batch_inputs.append(batch.x)
            batch_outputs.append(output)
            
            ### CALCULATE LOSS
            reconstruction_loss = mse_loss(batch.x.to(device), output[:,:node_features]) # compare first seq_len timesteps
            average_reconstruction_loss += reconstruction_loss.item()
            batch_loss = reconstruction_loss
            if args.predicted_timesteps > 0: 
                prediction_loss = mse_loss(batch.y.to(device), output[:,node_features:]) # compare last part to unseen data
                batch_loss += prediction_to_reconstruction_loss_ratio*prediction_loss
                average_prediction_loss += prediction_loss.item()

            ### BACKPROPAGATE
            batch_loss.backward()
            optimizer.step()
            average_loss += batch_loss.item()

            i += 1
            if (args.batch_limit > 0) and (i >= args.batch_limit): break # temporary -- for stopping training early
                
        inputs.append(torch.stack(batch_inputs))
        outputs.append(torch.stack(batch_outputs))
        
        average_loss = average_loss / i # use len(dataloader) for full batches
        average_reconstruction_loss = average_reconstruction_loss / i # use len(dataloader) for full batches
        average_prediction_loss = average_prediction_loss / i # use len(dataloader) for full batches

        losses.append(average_loss) 
        reconstruction_losses.append(average_reconstruction_loss)
        prediction_losses.append(average_prediction_loss)
        print("epoch : {}/{} | Loss = {:,.4f} | Reconstruction Loss: {:,.4f} | Prediction Loss: {:,.4f}, Time: {:.4f} sec".format(epoch+1, epochs, 
                                                                                                                average_loss,
                                                                                                                average_reconstruction_loss, 
                                                                                                                average_prediction_loss,
                                                                                                                time.time() - t),
                                                                                                                file=log)
        log.flush()
        
        if epoch == 0 and not checkpoint_loaded: best_loss = average_loss
        elif epoch == 0 and checkpoint_loaded: best_loss = min(average_loss, loss_checkpoint)
            
        if average_loss < best_loss:
            best_loss = average_loss
            torch.save({
             'epoch': epoch,
             'model_state_dict': model.state_dict(),
             'optimizer_state_dict': optimizer.state_dict(),
             'loss': best_loss,
             }, checkpoint_path)
            print("Better loss achieved -- saved model checkpoint to {}.".format(checkpoint_path), file=log)
            log.flush()
    return losses, reconstruction_losses, prediction_losses, inputs, outputs

losses, reconstruction_losses, prediction_losses, inputs, outputs = train(epochs=args.epochs)

loss_dict = {
	"overall_losses": losses,
	"reconstruction_losses": reconstruction_losses,
	"prediction_losses": prediction_losses,
			}

with open(os.path.join(save_folder,'losses.json'), 'w') as f:
    json.dump(loss_dict, f)

inputs = inputs[0].cpu().detach().numpy()
outputs = outputs[0].cpu().detach().numpy()

with open(os.path.join(save_folder,"train_inputs.npy"), "wb") as f:
    np.save(f, inputs)
with open(os.path.join(save_folder,"train_outputs.npy"), "wb") as f:
    np.save(f, outputs)

### MAKE PLOTS
fig, ax = plt.subplots(figsize=(8,6))
ax.plot(np.arange(len(losses)), reconstruction_losses, label="Reconstruction")
ax.set_xlabel("Epoch", fontsize=16)
ax.set_ylabel("Loss", fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
ax.legend(fontsize=14)
plt.savefig(os.path.join(save_folder,"losses.jpg"))

### LOOK AT PREDICTIONS
first_input_batch = inputs[0]
first_input_seq = first_input_batch[:n_joints, :]

# reshape to be n_joints x n_timesteps x n_dim
first_input_seq = first_input_seq.reshape((n_joints,args.seq_len,3))

first_predicted_batch = outputs[0]
first_predicted_seq = first_predicted_batch[:n_joints, :]

# reshape to be n_joints x n_timesteps x n_dim
first_predicted_seq = first_predicted_seq.reshape((n_joints,args.seq_len,3))

plt.figure(figsize=(10,7))
for joint in range(1): # first few joints
# for joint in range(first_seq.shape[0]): # all joints
    # plot x & y for the sequence
    plt.plot(first_input_seq[joint,:,0], first_input_seq[joint,:,1], 'o--', label="Input Joint "+str(joint)) 
    plt.plot(first_predicted_seq[joint,:,0], first_predicted_seq[joint,:,1], 'o--', label="Predicted Joint "+str(joint)) 
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=12)
plt.savefig(os.path.join(save_folder,"predictions_joint0.jpg"))
