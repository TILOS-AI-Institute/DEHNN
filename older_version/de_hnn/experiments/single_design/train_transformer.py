import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, Adagrad
import pickle
from torch import optim
from torch.utils.data import DataLoader
import numpy as np
import os
import time
import argparse
import scipy
import matplotlib.pyplot as plt

# For visualization
from utils import *

# Metrics
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score

# PyTorch geometric data loader
from torch_geometric.loader import DataLoader
from pyg_dataset import pyg_dataset

# Fix number of threads
torch.set_num_threads(4)

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
    parser.add_argument('--heads', '-heads', type = int, default = 4, help = 'Number of heads in Linear Transformer')
    parser.add_argument('--local_heads', '-local_heads', type = int, default = 1, help = 'Number of local attention heads in Linear Transformer')
    parser.add_argument('--depth', '-depth', type = int, default = 1, help = 'Depth in Linear Transformer')
    parser.add_argument('--pe_type', '-pe_type', type = str, default = 'none', help = 'Type of position encoding')
    parser.add_argument('--pe_dim', '-pe_dim', type = int, default = 5, help = 'Dimension of position encoding')
    parser.add_argument('--load_global_info', '-load_global_info', type = int, default = 0, help = 'Global information')
    parser.add_argument('--load_pd', '-load_pd', type = int, default = 0, help = 'Persistence diagram & Neighbor list')
    parser.add_argument('--test_mode', '-test_mode', type = int, default = 0, help = 'Test mode')
    parser.add_argument('--device', '-device', type = str, default = 'cpu', help = 'cuda/cpu')
    parser.add_argument('--graph_index', '-graph_index', type = int, default = 0, help = 'Index of the graph')
    parser.add_argument('--split', '-split', type = int, default = 1, help = 'Index of the split')
    args = parser.parse_args()
    return args

args = _parse_args()

if args.target == 'classify':
    import sys
    sys.path.insert(1, '../../models_inst_classify/')
    from linear_transformer_model import Linear_Transformer
    
else:
    import sys
    sys.path.insert(1, '../../models_inst/')
    from linear_transformer_model import Linear_Transformer

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

# Dataset
print(args.data_dir)

# Global information
load_global_info = False
if args.load_global_info == 1:
    load_global_info = True

load_pd = False
if args.load_pd == 1:
    load_pd = True

pe_type = args.pe_type
pe_dim = args.pe_dim

if pe_type == 'lap':
    dataset = pyg_dataset(data_dir = args.data_dir, graph_index = args.graph_index, target = args.target, load_pe = True, num_eigen = pe_dim, load_global_info = load_global_info, load_pd = load_pd, concat = True, split = args.split)
else:
    dataset = pyg_dataset(data_dir = args.data_dir, graph_index = args.graph_index, target = args.target, load_global_info = load_global_info, load_pd = load_pd, concat = True, split = args.split)

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
    if args.target == 'classify':
        num_outputs = 2#data.y.size(1)
    else:
        num_outputs = data.y.size(1)
    break

print('Number of node features:', node_dim)
print('Number of edge features:', edge_dim)
print('Number of outputs:', num_outputs)

if pe_type == 'lap':
    node_dim += pe_dim

    print('Number of eigenvectors:', pe_dim)
    print('Number of node features + eigenvectors:', node_dim)

# Search for the maximum length
max_size = 0

for batch_idx, data in enumerate(dataloader):
    num_nodes = data.x.size(0)
    if num_nodes > max_size:
        max_size = num_nodes

max_seq_len = 1
while max_seq_len < max_size:
    max_seq_len *= 2

print('Maximum graph size:', max_size)
print('Maximum sequence length (for Transformer):', max_seq_len)

# Init model and optimizer
model = Linear_Transformer(input_dim = node_dim, hidden_dim = args.hidden_dim, output_dim = num_outputs, max_seq_len = max_seq_len, heads = args.heads, depth = args.depth, n_local_attn_heads = args.local_heads).to(device = device)
optimizer = Adagrad(model.parameters(), lr = args.learning_rate)

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
        data = data.to(device = device)
        targets = data.y
        node_feat = data.x

        # Position encoding
        if pe_type == 'lap':
            node_feat = torch.cat([node_feat, data.evects], dim = 1)

        # Batch dimension
        node_feat = torch.unsqueeze(node_feat, dim = 0)
        targets = torch.unsqueeze(targets, dim = 0)

        # Model
        predict = model(node_feat)

        # Train indices
        predict = predict[:, dataset.train_indices, :]
        targets = targets[:, dataset.train_indices, :]

        optimizer.zero_grad()
        
        # Mean squared error loss
        targets = targets.contiguous()
        predict = predict.contiguous()
        #loss = F.mse_loss(predict.view(-1), targets.view(-1), reduction = 'mean')
        loss = F.nll_loss(predict[0], targets[0].view(-1))

        sum_error += loss.item()#torch.sum(torch.abs(predict.view(-1) - target.view(-1))).detach().cpu().numpy()
        num_samples += targets.size(1)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        nBatch += 1
        if batch_idx % 100 == 0:
            print('Batch', batch_idx, '/', len(dataloader),': Loss =', loss.item())
            LOG.write('Batch ' + str(batch_idx) + '/' + str(len(dataloader)) + ': Loss = ' + str(loss.item()) + '\n')

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
            data = data.to(device = device)
            node_feat = data.x
            targets = data.y

            # Position encoding
            if pe_type == 'lap':
                node_feat = torch.cat([node_feat, data.evects], dim = 1)

            # Batch dimension
            node_feat = torch.unsqueeze(node_feat, dim = 0)
            targets = torch.unsqueeze(targets, dim = 0)

            # Model
            predict = model(node_feat)
            
            # Validation indices
            predict = predict[:, dataset.valid_indices, :]
            targets = targets[:, dataset.valid_indices, :]

            # Mean squared error loss
            targets = targets.contiguous()
            predict = predict.contiguous()
            
            loss = F.nll_loss(predict[0], targets[0].view(-1))
            
            #loss = F.mse_loss(predict.view(-1), targets.view(-1), reduction = 'mean')

            total_loss += loss.item()
            nBatch += 1

            # Mean average error
            sum_error += loss.item()#torch.sum(torch.abs(predict.view(-1) - target.view(-1))).detach().cpu().numpy()
            num_samples += targets.size(1)
             
            if batch_idx % 100 == 0:
                print('Valid Batch', batch_idx, '/', len(dataloader),': Loss =', loss.item())
                LOG.write('Valid Batch ' + str(batch_idx) + '/' + str(len(dataloader)) + ': Loss = ' + str(loss.item()) + '\n')

    valid_mae = sum_error #/ (num_samples * num_outputs)
    avg_loss = total_loss / nBatch

    print('Valid average loss:', avg_loss)
    LOG.write('Valid average loss: ' + str(avg_loss) + '\n')
    print('Valid MAE:', valid_mae)
    LOG.write('Valid MAE: ' + str(valid_mae) + '\n')
    print("Valid time =", "{:.5f}".format(time.time() - t))
    LOG.write("Valid time = " + "{:.5f}".format(time.time() - t) + "\n")
    
    if valid_mae < best_mae:
        best_mae = valid_mae
        print('Current best MAE updated:', best_mae)
        LOG.write('Current best MAE updated: ' + str(best_mae) + '\n')
        
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
        data = data.to(device = device)
        node_feat = data.x
        targets = data.y

        # Position encoding
        if pe_type == 'lap':
            node_feat = torch.cat([node_feat, data.evects], dim = 1)

        # Batch dimension
        node_feat = torch.unsqueeze(node_feat, dim = 0)
        targets = torch.unsqueeze(targets, dim = 0)

        # Model
        predict = model(node_feat)
        
        # Test indices
        predict = predict[:, dataset.test_indices, :]
        targets = targets[:, dataset.test_indices, :]
        
        # Mean squared error loss
        targets = targets.contiguous()
        predict = predict.contiguous()
        #loss = F.mse_loss(predict.view(-1), targets.view(-1), reduction = 'mean')
        loss = F.nll_loss(predict[0], targets[0].view(-1))
        y_test.append(targets.view(-1))
        y_hat.append(predict.view(-1))

        total_loss += loss.item()
        nBatch += 1

        sum_error += loss.item()#torch.sum(torch.abs(predict.view(-1) - target.view(-1))).detach().cpu().numpy()
        num_samples += targets.size(1)

        if batch_idx % 100 == 0:
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

# Visualiation
truth = torch.cat(y_test, dim = 0).cpu().detach().numpy()
predict = torch.cat(y_hat, dim = 0).cpu().detach().numpy()

with open(args.dir + "/" + args.name + ".truth.npy", 'wb') as f:
    np.save(f, truth)

with open(args.dir + "/" + args.name + ".predict.npy", 'wb') as f:
    np.save(f, predict)

method_name = "Transformer"
design_name = dataset.design_name
output_name = args.dir + "/" + args.name + ".png"
#plot_figure(truth, predict, method_name, design_name, output_name)

LOG.close()
print('Done')

