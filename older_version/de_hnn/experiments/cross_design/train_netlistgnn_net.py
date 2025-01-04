import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, Adagrad
from torch_geometric.utils import degree
from torch import optim

import numpy as np
import pickle
import os
import time
import argparse
import scipy
from tqdm import tqdm

# For visualization
from utils import *

# PyTorch geometric data loader
from torch_geometric.loader import DataLoader
from pyg_dataset_net import pyg_dataset

# NetlistGNN data loader
from load_netlistgnn_data import load_data

# Deep Graph Library (DGL)
import dgl

# To convert from our data to DGL
from convert_to_dgl_net import *

# Model
import sys
sys.path.insert(1, '../../models_net/')
from NetlistGNN import * # NetlistGNN from PKU

# Fix number of threads
torch.set_num_threads(4)

def _parse_args():
    parser = argparse.ArgumentParser(description = 'Supervised learning')
    parser.add_argument('--dir', '-dir', type = str, default = '.', help = 'Directory to save results')
    parser.add_argument('--target', '-target', type = str, default = 'none', help = 'Learning target')
    parser.add_argument('--data_dir', '-data_dir', type = str, default = '.', help = 'Directory that contains all the raw datasets')
    parser.add_argument('--name', '-name', type = str, default = 'NAME', help = 'Name')
    parser.add_argument('--num_epoch', '-num_epoch', type = int, default = 2048, help = 'Number of epochs')
    parser.add_argument('--batch_size', '-batch_size', type = int, default = 20, help = 'Batch size')
    parser.add_argument('--learning_rate', '-learning_rate', type = float, default = 0.001, help = 'Initial learning rate')
    parser.add_argument('--seed', '-s', type = int, default = 123456789, help = 'Random seed')
    parser.add_argument('--test_mode', '-test_mode', type = int, default = 0, help = 'Test mode')
    parser.add_argument('--pe', '-pe', type = str, default = 'none', help = 'Position encoding')
    parser.add_argument('--pos_dim', '-pos_dim', type = int, default = 0, help = 'Dimension of position encoding')
    parser.add_argument('--load_global_info', '-load_global_info', type = int, default = 0, help = 'Global information')
    parser.add_argument('--load_pd', '-load_pd', type = int, default = 0, help = 'Persistence diagram & Neighbor list')
    parser.add_argument('--fold', '-fold', type = int, default = 0, help = 'Fold index in cross-validation')
    parser.add_argument('--device', '-device', type = str, default = 'cpu', help = 'cuda/cpu')
    
    # For NetlistGNN
    parser.add_argument('--n_layer', '-n_layer', type = int, default = 2, help = 'Number of layers')
    parser.add_argument('--node_feats', '-node_feats', type = int, default = 64, help = 'Number of node features')
    parser.add_argument('--net_feats', '-net_feats', type = int, default = 128, help = 'Number of net features')
    parser.add_argument('--pin_feats', '-pin_feats', type = int, default = 16, help = 'Number of pin features')
    parser.add_argument('--edge_feats', '-edge_feats', type = int, default = 4, help = 'Number of edge features')
    parser.add_argument('--outtype', "--outtype", type = str, default = 'relu', help = 'Check NetlistGNN for details')
    parser.add_argument('--recurrent', "--recurrent", type = bool, default = False, help = 'Check NetlistGNN for details')
    parser.add_argument('--topo_conv_type', type = str, default = 'CFCNN', help = 'Check NetlistGNN for details')
    parser.add_argument('--geom_conv_type', type = str, default = 'SAGE', help = 'Check NetlistGNN for details')

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

sparse = False
if sparse:
    from pyg_dataset_sparse import *

# Dataset
print(args.dir)

pe = args.pe
pos_dim = args.pos_dim

load_global_info = False
if args.load_global_info == 1:
    load_global_info = True

load_pd = False
if args.load_pd == 1:
    load_pd = True

if pe == 'lap':
    train_dataset = pyg_dataset(data_dir = args.data_dir, fold_index = args.fold, split = 'train', target = args.target, load_pe = True, num_eigen = pos_dim, load_global_info = load_global_info, load_pd = load_pd)
    valid_dataset = pyg_dataset(data_dir = args.data_dir, fold_index = args.fold, split = 'valid', target = args.target, load_pe = True, num_eigen = pos_dim, load_global_info = load_global_info, load_pd = load_pd)
    test_dataset = pyg_dataset(data_dir = args.data_dir, fold_index = args.fold, split = 'test', target = args.target, load_pe = True, num_eigen = pos_dim, load_global_info = load_global_info, load_pd = load_pd)
else:
    train_dataset = pyg_dataset(data_dir = args.data_dir, fold_index = args.fold, split = 'train', target = args.target, load_global_info = load_global_info, load_pd = load_pd, load_pe = False)
    valid_dataset = pyg_dataset(data_dir = args.data_dir, fold_index = args.fold, split = 'valid', target = args.target, load_global_info = load_global_info, load_pd = load_pd, load_pe = False)
    test_dataset = pyg_dataset(data_dir = args.data_dir, fold_index = args.fold, split = 'test', target = args.target, load_global_info = load_global_info, load_pd = load_pd, load_pe = False)
    
# Data loaders
batch_size = args.batch_size
print(batch_size)
train_dataloader = DataLoader(train_dataset, batch_size, shuffle = True)
valid_dataloader = DataLoader(valid_dataset, batch_size, shuffle = False)
test_dataloader = DataLoader(test_dataset, batch_size, shuffle = False)

print('Number of training examples:', len(train_dataset))
print('Number of testing examples:', len(test_dataset))

for batch_idx, pyg_data in enumerate(train_dataloader):
    print(batch_idx)
    print(pyg_data)

    # Number of outputs
    num_outputs = pyg_data.y.size(1)

    # Conversion to DGL format
    dgl_data = convert_to_dgl(pyg_data)

    in_node_feats = dgl_data.nodes['node'].data['hv'].shape[1]
    in_net_feats = dgl_data.nodes['net'].data['hv'].shape[1]
    in_pin_feats = dgl_data.edges['pinned'].data['he'].shape[1]
    in_edge_feats = dgl_data.edges['near'].data['he'].shape[1]

    break

print("Number of outputs:", num_outputs)
print("Input node features:", in_node_feats)
print("Input net features:", in_net_feats)
print("Input pin features:", in_pin_feats)
print("Input edge features:", in_edge_feats)

# Statistics
y = []
for batch_idx, data in enumerate(train_dataloader):
    y.append(data.y.detach().numpy())
y = np.concatenate(y)

y_min = np.min(y)
y_max = np.max(y)
y_mean = np.mean(y)
y_std = np.std(y)

print('y min:', y_min)
print('y max:', y_max)
print('y mean:', y_mean)
print('y std:', y_std)

# Init the NetlistGNN model and optimizer
config = {
    'N_LAYER': args.n_layer,
    'NODE_FEATS': args.node_feats,
    'NET_FEATS': args.net_feats,
    'PIN_FEATS': args.pin_feats,
    'EDGE_FEATS': args.edge_feats,
}

model = NetlistGNN(
    in_node_feats = in_node_feats,
    in_net_feats = in_net_feats,
    in_pin_feats = in_pin_feats,
    in_edge_feats = in_edge_feats,
    n_target = 1,
    activation = args.outtype,
    config = config,
    recurrent = args.recurrent,
    topo_conv_type = args.topo_conv_type,
    geom_conv_type = args.geom_conv_type
).to(device)

optimizer = Adagrad(model.parameters(), lr = args.learning_rate)

num_parameters = sum(param.numel() for param in model.parameters() if param.requires_grad)
print('Number of learnable parameters:', num_parameters)
LOG.write('Number of learnable parameters: ' + str(num_parameters) + '\n')
print('Done with model creation')

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
    
    for batch_idx, data in enumerate(train_dataloader):
        target = (data.y - y_mean) / y_std
        target = target.to(device)

        # Convert from PyG to DGL
        dgl_data = convert_to_dgl(data)

        # NetlistGNN
        predict_instance, predict_net = model.forward(
                in_node_feat = dgl_data.nodes['node'].data['hv'].to(device),
                in_net_feat = dgl_data.nodes['net'].data['hv'].to(device),
                in_pin_feat = dgl_data.edges['pinned'].data['he'].to(device),
                in_edge_feat = dgl_data.edges['near'].data['he'].to(device),
                node_net_graph = dgl_data.to(device),
        )
        
        #predict = torch.cat([predict_instance, predict_net], dim = 0)
        predict = predict_net
        
        optimizer.zero_grad()

        # Mean squared error loss
        loss = F.mse_loss(predict.view(-1), target.view(-1), reduction = 'mean')
 
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        nBatch += 1

        sum_error += torch.sum(torch.abs(predict.view(-1) - target.view(-1))).detach().cpu().numpy()
        num_samples += predict.size(0)

        if batch_idx % 10 == 0:
            print('Batch', batch_idx, '/', len(train_dataloader),': Loss =', loss.item())
            LOG.write('Batch ' + str(batch_idx) + '/' + str(len(train_dataloader)) + ': Loss = ' + str(loss.item()) + '\n')

    train_mae = sum_error / (num_samples * num_outputs)
    avg_loss = total_loss / nBatch

    print('Train average loss:', avg_loss)
    LOG.write('Train average loss: ' + str(avg_loss) + '\n')
    print('Train MAE:', train_mae)
    LOG.write('Train MAE: ' + str(train_mae) + '\n')
    print('Train MAE (original scale):', train_mae * y_std)
    LOG.write('Train MAE (original scale): ' + str(train_mae * y_std) + '\n')
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
        for batch_idx, data in enumerate(valid_dataloader):
            target = (data.y - y_mean) / y_std
            target = target.to(device)

            # Convert from PyG to DGL
            dgl_data = convert_to_dgl(data)

            # NetlistGNN
            predict_instance, predict_net = model.forward(
                    in_node_feat = dgl_data.nodes['node'].data['hv'].to(device),
                    in_net_feat = dgl_data.nodes['net'].data['hv'].to(device),
                    in_pin_feat = dgl_data.edges['pinned'].data['he'].to(device),
                    in_edge_feat = dgl_data.edges['near'].data['he'].to(device),
                    node_net_graph = dgl_data.to(device),
            )
            
            predict = predict_net

            # Mean squared error loss
            loss = F.mse_loss(predict.view(-1), target.view(-1), reduction = 'mean')

            total_loss += loss.item()
            nBatch += 1

            sum_error += torch.sum(torch.abs(predict.view(-1) - target.view(-1))).detach().cpu().numpy()
            num_samples += predict.size(0)
             
            if batch_idx % 10 == 0:
                print('Valid Batch', batch_idx, '/', len(valid_dataloader),': Loss =', loss.item())
                LOG.write('Valid Batch ' + str(batch_idx) + '/' + str(len(valid_dataloader)) + ': Loss = ' + str(loss.item()) + '\n')

    valid_mae = sum_error / (num_samples * num_outputs)
    avg_loss = total_loss / nBatch

    print('Valid average loss:', avg_loss)
    LOG.write('Valid average loss: ' + str(avg_loss) + '\n')
    print('Valid MAE:', valid_mae)
    LOG.write('Valid MAE: ' + str(valid_mae) + '\n')
    print('Valid MAE (original scale):', valid_mae * y_std)
    LOG.write('Valid MAE (original scale): ' + str(valid_mae * y_std) + '\n')
    print("Valid time =", "{:.5f}".format(time.time() - t))
    LOG.write("Valid time = " + "{:.5f}".format(time.time() - t) + "\n")

    if valid_mae < best_mae:
        best_mae = valid_mae
        print('Current best MAE updated:', best_mae)
        LOG.write('Current best MAE updated: ' + str(best_mae) + '\n')
        print('Current best MAE (original scale) updated:', best_mae * y_std)
        LOG.write('Current best MAE (original scale) updated: ' + str(best_mae * y_std) + '\n')
        
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
    print('Best valid MAE (original scale):', best_mae * y_std)
    LOG.write('Best valid MAE (original scale): ' + str(best_mae * y_std) + '\n')

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
    for batch_idx, data in enumerate(test_dataloader):
        target = (data.y - y_mean) / y_std
        target = target.to(device)

        # Convert from PyG to DGL
        dgl_data = convert_to_dgl(data)

        # NetlistGNN
        predict_instance, predict_net = model.forward(
                in_node_feat = dgl_data.nodes['node'].data['hv'].to(device),
                in_net_feat = dgl_data.nodes['net'].data['hv'].to(device),
                in_pin_feat = dgl_data.edges['pinned'].data['he'].to(device),
                in_edge_feat = dgl_data.edges['near'].data['he'].to(device),
                node_net_graph = dgl_data.to(device),
        )
        
        predict = predict_net

        # Mean squared error loss
        loss = F.mse_loss(predict.view(-1), target.view(-1), reduction = 'mean')

        y_test.append(target.view(-1))
        y_hat.append(predict.view(-1))

        total_loss += loss.item()
        nBatch += 1

        sum_error += torch.sum(torch.abs(predict.view(-1) - target.view(-1))).detach().cpu().numpy()
        num_samples += predict.size(0)

        if batch_idx % 10 == 0:
            print('Test Batch', batch_idx, '/', len(test_dataloader),': Loss =', loss.item())
            LOG.write('Test Batch ' + str(batch_idx) + '/' + str(len(test_dataloader)) + ': Loss = ' + str(loss.item()) + '\n')

test_mae = sum_error / (num_samples * num_outputs)
avg_loss = total_loss / nBatch

print('--------------------------------------')
LOG.write('--------------------------------------\n')
print('Test average loss:', avg_loss)
LOG.write('Test average loss: ' + str(avg_loss) + '\n')
print('Test MAE:', test_mae)
LOG.write('Test MAE: ' + str(test_mae) + '\n')
print('Test MAE (original scale):', test_mae * y_std)
LOG.write('Test MAE (original scale): ' + str(test_mae * y_std) + '\n')
print("Test time =", "{:.5f}".format(time.time() - t))
LOG.write("Test time = " + "{:.5f}".format(time.time() - t) + "\n")


truth = torch.cat(y_test, dim = 0).cpu().detach().numpy() * y_std + y_mean
predict = torch.cat(y_hat, dim = 0).cpu().detach().numpy() * y_std + y_mean

with open(args.dir + "/" + args.name + ".truth.npy", 'wb') as f:
    np.save(f, truth)

with open(args.dir + "/" + args.name + ".predict.npy", 'wb') as f:
    np.save(f, predict)

# method_name = "NetlistGNN - PKU"
# design_name = designs_list[args.fold]
# output_name = args.dir + "/" + args.name + ".png"

# plot_figure(truth, predict, method_name, design_name, output_name)
print('Done')

LOG.close()
