import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, Adagrad
from torch import optim

import numpy as np
import os
import time
import argparse
import scipy
import pickle

# For visualization
from utils import *

# PyTorch geometric data loader
from torch_geometric.loader import DataLoader
from pyg_dataset import pyg_dataset

# Model
import sys
sys.path.insert(1, '../../models_inst_classify/')
from mlp_model import MLP

# Fix number of threads
torch.set_num_threads(4)

def weighted_mse(input, target, weight):
    return (weight * (input - target) ** 2).mean()

def _parse_args():
    parser = argparse.ArgumentParser(description = 'Supervised learning')
    parser.add_argument('--dir', '-dir', type = str, default = '.', help = 'Directory')
    parser.add_argument('--target', '-target', type = str, default = 'none', help = 'Learning target')
    parser.add_argument('--data_dir', '-data_dir', type = str, default = '.', help = 'Directory that contains the raw datasets')
    parser.add_argument('--name', '-name', type = str, default = 'NAME', help = 'Name')
    parser.add_argument('--num_epoch', '-num_epoch', type = int, default = 2048, help = 'Number of epochs')
    parser.add_argument('--batch_size', '-batch_size', type = int, default = 20, help = 'Batch size')
    parser.add_argument('--learning_rate', '-learning_rate', type = float, default = 0.001, help = 'Initial learning rate')
    parser.add_argument('--seed', '-s', type = int, default = 123456789, help = 'Random seed')
    parser.add_argument('--hidden_dim', '-hidden_dim', type = int, default = 32, help = 'Hidden dimension')
    parser.add_argument('--graph_index', '-graph_index', type = int, default = 0, help = 'Index of graph')
    parser.add_argument('--load_pe', '-load_pe', type = int, default = 0, help = 'Position encoding')
    parser.add_argument('--load_global_info', '-load_global_info', type = int, default = 0, help = 'Global information')
    parser.add_argument('--load_pd', '-load_pd', type = int, default = 0, help = 'Persistence diagram & Neighbor list')
    parser.add_argument('--num_eigen', '-num_eigen', type = int, default = 0, help = 'Number of eigenvectors')
    parser.add_argument('--test_mode', '-test_mode', type = int, default = 0, help = 'Test mode')
    parser.add_argument('--device', '-device', type = str, default = 'cpu', help = 'cuda/cpu')
    args = parser.parse_args()
    return args

args = _parse_args()
if args.test_mode == 0:
    log_name = args.dir + "/" + args.name + ".log"
else:
    print("Test mode")
    log_name = args.dir + "/" + args.name + ".test_mode.log"
model_name = args.dir + "/" + args.name + ".model"
LOG = open(log_name, "w")

# Fix CPU torch random seed
torch.manual_seed(args.seed)

# Fix GPU torch random seed
torch.cuda.manual_seed(args.seed)

# Fix the Numpy random seed
np.random.seed(args.seed)

# Train on CPU (hide GPU) due to memory constraints
# os.environ['CUDA_VISIBLE_DEVICES'] = ""
device = args.device
print(device)

# Position encoding
if args.load_pe == 1:
    load_pe = True
    num_eigen = args.num_eigen

# Global information
load_global_info = False
if args.load_global_info == 1:
    load_global_info = True

load_pd = False
if args.load_pd == 1:
    load_pd = True

# Dataset
print(args.data_dir)
if load_pe:
    dataset = pyg_dataset(data_dir = args.data_dir, graph_index = args.graph_index, target = args.target, load_pe = True, num_eigen = num_eigen, load_global_info = load_global_info, load_pd = load_pd, concat = False)
else:
    dataset = pyg_dataset(data_dir = args.data_dir, graph_index = args.graph_index, target = args.target, load_global_info = load_global_info, load_pd = load_pd, concat = False)

# Data loaders
batch_size = args.batch_size
dataloader = DataLoader(dataset, batch_size, shuffle = False)

print('Number of nodes in the training set:', dataset.train_indices.shape[0])
print('Number of nodes in the validation set:', dataset.valid_indices.shape[0])
print('Number of nodes in the testing set:', dataset.test_indices.shape[0])

for batch_idx, data in enumerate(dataloader):
    print(batch_idx)
    print(data)
    node_dim = data.x.size(1)
    edge_dim = data.edge_attr.size(1)
    num_outputs = 2#data.y.size(1)

print('Number of node features:', node_dim)
print('Number of edge features:', edge_dim)
print('Number of outputs:', num_outputs)

if load_pe == True:
    node_dim += num_eigen
    
    print('Number of eigenvectors:', num_eigen)
    print('Number of node features + eigenvectors:', node_dim)

# Init model and optimizer
model = MLP(input_dim = node_dim, hidden_dim = args.hidden_dim, output_dim = num_outputs).to(device = device)
optimizer = Adagrad(model.parameters(), lr = args.learning_rate)

print(model)

num_parameters = sum(param.numel() for param in model.parameters() if param.requires_grad)
print('Number of learnable parameters:', num_parameters)
LOG.write('Number of learnable parameters: ' + str(num_parameters) + '\n')

# Test mode
if args.test_mode == 1:
    print("Skip the training")
    num_epoch = 0
else:
    num_epoch = args.num_epoch

# Train model
best_mae = 1e9
for epoch in range(num_epoch):
    print('--------------------------------------')
    print('Epoch', epoch)
    LOG.write('--------------------------------------\n')
    LOG.write('Epoch ' + str(epoch) + '\n')

    # Training
    t = time.time()
    total_loss = 0.0
    nBatch = 0
    sum_error = 0.0
    num_samples = 0
    
    for batch_idx, data in enumerate(dataloader):
        if load_pe == True:
            node_feat = torch.cat([data.x, data.evects[:data.x.shape[0]]], dim = 1).to(device = device)
        else:
            node_feat = data.x.to(device = device)
        targets = data.y.to(device = device)
        #weights = data.weights.to(device = device)
        
        # Model
        predict = model(node_feat)
         
        # Train indices
        predict = predict[dataset.train_indices, :]
        targets = targets[dataset.train_indices, :]
        #weights = weights[dataset.train_indices]

        optimizer.zero_grad()
        
        # Mean squared error loss

        print(predict.shape, targets.shape)

        loss = F.nll_loss(predict, targets.view(-1))
        mse_loss = loss

        loss.backward()
        optimizer.step()

        sum_error += loss.item()#torch.sum(torch.abs(predict.view(-1) - target.view(-1))).detach().cpu().numpy()
        num_samples += predict.size(0)
        
        total_loss += loss.item()
        nBatch += 1
        if batch_idx % 10 == 0:
            print('Batch', batch_idx, '/', len(dataloader),': Loss =', loss.item())
            LOG.write('Batch ' + str(batch_idx) + '/' + str(len(dataloader)) + ': Loss = ' + str(loss.item()) + ' MSE loss = ' + str(mse_loss.item()) + '\n')

    train_mae = sum_error #/ (num_samples * num_outputs)
    avg_loss = total_loss / nBatch

    print('Train average loss:', avg_loss)
    LOG.write('Train average loss: ' + str(avg_loss) + '\n')
    print('Train MAE:', train_mae)
    LOG.write('Train MAE: ' + str(train_mae) + '\n')
    print("Train time =", "{:.5f}".format(time.time() - t))
    LOG.write("Train time = " + "{:.5f}".format(time.time() - t) + "\n")

    # Validation
    t = time.time()
    model.eval()
    total_loss = 0.0
    nBatch = 0

    with torch.no_grad():
        sum_error = 0.0
        num_samples = 0
        for batch_idx, data in enumerate(dataloader):
            if load_pe == True:
                node_feat = torch.cat([data.x, data.evects[:data.x.shape[0]]], dim = 1).to(device = device)
            else:
                node_feat = data.x.to(device = device)
            targets = data.y.to(device = device)
            #weights = data.weights.to(device = device)
            # Model
            predict = model(node_feat)
            
            # Validation indices
            predict = predict[dataset.valid_indices, :]
            targets = targets[dataset.valid_indices, :]
            #weights = weights[dataset.valid_indices]

            # Mean squared error loss
            loss = F.nll_loss(predict, targets.view(-1))
            mse_loss = loss#F.mse_loss(predict.view(-1), target.view(-1), reduction = 'mean')

            total_loss += loss.item()
            nBatch += 1

            # Mean average error
            sum_error += loss.item()#torch.sum(torch.abs(predict.view(-1) - target.view(-1))).detach().cpu().numpy()
            num_samples += predict.size(0)
             
            if batch_idx % 10 == 0:
                print('Valid Batch', batch_idx, '/', len(dataloader),': Loss =', loss.item())
                LOG.write('Valid Batch ' + str(batch_idx) + '/' + str(len(dataloader)) + ': Loss = ' + str(loss.item()) + ' MSE loss = ' + str(mse_loss.item()) + '\n')

    valid_mae = sum_error #/ num_samples
    avg_loss = total_loss / nBatch
    print('Valid average loss:', avg_loss)
    LOG.write('Valid average loss: ' + str(avg_loss) + '\n')
    print('Valid MAE:', valid_mae)
    LOG.write('Valid MAE: ' + str(valid_mae) + '\n')
    print("Valid time =", "{:.5f}".format(time.time() - t))
    LOG.write("Valid time = " + "{:.5f}".format(time.time() - t) + "\n")
    
    
    if avg_loss < best_mae:
        best_mae = avg_loss
        print('Current best loss updated:', best_mae)
        LOG.write('Current best loss updated: ' + str(best_mae) + '\n')
        
        torch.save(model.state_dict(), model_name)
        print("Save the best model to " + model_name)
        LOG.write("Save the best model to " + model_name + "\n")
    else:
        # Early stopping
        # break
        pass

if args.test_mode == 0:
    print('--------------------------------------')
    LOG.write('--------------------------------------\n')
    print('Best valid MAE:', best_mae)
    LOG.write('Best valid MAE: ' + str(best_mae) + '\n')

# Load the model with the best validation
print("Load the trained model at", model_name)
model.load_state_dict(torch.load(model_name))

# Testing
t = time.time()
model.eval()
total_loss = 0.0
nBatch = 0

y_test = []
y_hat = []

with torch.no_grad():
    sum_error = 0.0
    num_samples = 0
    for batch_idx, data in enumerate(dataloader):
        if load_pe == True:
            node_feat = torch.cat([data.x, data.evects[:data.x.shape[0]]], dim = 1).to(device = device)
        else:
            node_feat = data.x.to(device = device)
        targets = data.y.to(device = device)
        #weights = data.weights.to(device = device)
        # Model
        predict = model(node_feat)
        
        # Test indices
        predict = predict[dataset.test_indices, :]
        targets = targets[dataset.test_indices, :]
        #weights = weights[dataset.test_indices]
        

        # Mean squared error loss
        loss = F.nll_loss(predict, targets.view(-1), reduction = 'mean')

        total_loss += loss.item()
        nBatch += 1

        y_test.append(targets.view(-1))
        y_hat.append(predict.view(-1))

        sum_error += loss.item()#torch.sum(torch.abs(predict.view(-1) - target.view(-1))).detach().cpu().numpy()
        num_samples += predict.size(0)

        if batch_idx % 10 == 0:
            print('Test Batch', batch_idx, '/', len(dataloader),': Loss =', loss.item())
            LOG.write('Test Batch ' + str(batch_idx) + '/' + str(len(dataloader)) + ': Loss = ' + str(loss.item()) + '\n')

test_mae = sum_error #/ (num_samples * num_outputs)
avg_loss = total_loss / nBatch

print('--------------------------------------')
LOG.write('--------------------------------------\n')
print('Test average loss:', avg_loss)
LOG.write('Test average loss: ' + str(avg_loss) + '\n')
print('Test MAE:', test_mae)
LOG.write('Test MAE: ' + str(test_mae) + '\n')
print("Test time =", "{:.5f}".format(time.time() - t))
LOG.write("Test time = " + "{:.5f}".format(time.time() - t) + "\n")

# # Visualiation
truth = torch.concat(y_test, dim=0).cpu().detach().numpy() 
predict = torch.concat(y_hat, dim=0).cpu().detach().numpy() 
#truth = y_test#.cpu().detach().numpy()
#predict = y_hat#.cpu().detach().numpy()

with open(args.dir + "/" + args.name + ".truth.npy", 'wb') as f:
    np.save(f, truth)

with open(args.dir + "/" + args.name + ".predict.npy", 'wb') as f:
    np.save(f, predict)

# method_name = 'MLP'
# design_name = dataset.design_name
# output_name = args.dir + "/" + args.name + ".png"

# plot_figure(truth, predict, method_name, design_name, output_name)
# print('Done')

LOG.close()
