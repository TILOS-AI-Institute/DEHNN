import os
import numpy as np
import pickle
import torch
import torch.nn
from torch_geometric.data import Dataset
from torch_geometric.data import Data, HeteroData
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from utils import *
import torch.nn as nn
import torch.optim as optim
import wandb
from tqdm import tqdm

import sys
sys.path.insert(1, 'data/')

from pyg_dataset import NetlistDataset

sys.path.append("models/layers/")
from models.model import GNN_node

test = False
restart = False
debug_mode = False

if debug_mode:
    load_indices = [0, 1, 2, 3, 4]
else:
    load_indices = None

dataset = NetlistDataset(data_dir="data/processed_datasets", load_pe = True, pl = True, processed = True, load_indices = load_indices)

h_dataset = []

for data in tqdm(dataset):
    num_instances = data.node_congestion.shape[0]
    data.num_instances = num_instances
    data.edge_index_source_to_net[1] = data.edge_index_source_to_net[1] - num_instances
    data.edge_index_sink_to_net[1] = data.edge_index_sink_to_net[1] - num_instances

    h_data = HeteroData()
    h_data['node'].x = data.node_features
    h_data['node'].y = data.node_congestion
    
    h_data['net'].x = data.net_features
    h_data['net'].y = data.net_hpwl
    
    h_data['node', 'as_a_sink_of', 'net'].edge_index, h_data['node', 'as_a_sink_of', 'net'].edge_weight = gcn_norm(data.edge_index_sink_to_net, add_self_loops=False)
    
    h_data['node', 'as_a_source_of', 'net'].edge_index = data.edge_index_source_to_net

    h_data.batch = data.batch
    h_data.num_vn = data.num_vn
    h_data.num_instances = num_instances
    h_dataset.append(h_data)

sys.path.append("models/layers/")
device = "cuda"

model = GNN_node(4, 32, 8, 1, node_dim = data.node_features.shape[1], net_dim = data.net_features.shape[1]).to(device)

if restart:
    model = torch.load("best_dehnn_model.pt")
    
criterion_node = nn.CrossEntropyLoss()
criterion_net = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

all_train_indices, all_valid_indices, all_test_indices = pickle.load(open("data/cross_design_data_split.pt", "rb"))

best_total_val = None

if not test:
    if not debug_mode:
        for epoch in range(100):
            np.random.shuffle(all_train_indices)
        
            loss_node_all = 0
            loss_net_all = 0
            val_loss_node_all = 0
            val_loss_net_all = 0
            
            model.train()
            all_train_idx = 0
            for data in tqdm([h_dataset[idx] for idx in all_train_indices]):
                try:
                    optimizer.zero_grad()
                    node_representation, net_representation = model(data, device)
                    loss_node = criterion_node(node_representation, data['node'].y.to(device))
                    loss_net = criterion_net(net_representation.flatten(), data['net'].y.to(device))
                    loss = loss_node + 0.001*loss_net
                    loss.backward()
                    optimizer.step()
                except:
                    print("OOM")
                    continue
            
                loss_node_all += loss_node.item()
                loss_net_all += loss_net.item()
                all_train_idx += 1
    
            print(loss_node_all/all_train_idx, loss_net_all/all_train_idx)
            wandb.log({
                "loss_node": loss_node_all/all_train_idx,
                "loss_net": loss_net_all/all_train_idx,
            })
        
            model.eval()
            all_valid_idx = 0
            for data in tqdm([h_dataset[idx] for idx in all_valid_indices]):
                try:
                    node_representation, net_representation = model(data, device)
                    val_loss_node = criterion_node(node_representation, data['node'].y.to(device))
                    val_loss_net = criterion_net(net_representation.flatten(), data['net'].y.to(device))
                    val_loss_node_all += val_loss_node.item()
                    val_loss_net_all += val_loss_net.item()
                    all_valid_idx += 1
                except:
                    print("OOM")
                    continue
            
            if (best_total_val is None) or ((val_loss_node_all/all_valid_idx) < best_total_val):
                best_total_val = val_loss_node_all/all_valid_idx
                torch.save(model, "best_dehnn_model.pt")
    
            print(val_loss_node_all/all_valid_idx, val_loss_net_all/all_valid_idx)
            wandb.log({
                "val_loss_node": val_loss_node_all/all_valid_idx,
                "val_loss_net": val_loss_net_all/all_valid_idx,
            })
    else:
        for data in tqdm(h_dataset):
            optimizer.zero_grad()
            node_representation, net_representation = model(data, device)
            loss_node = criterion_node(node_representation, data['node'].y.to(device))
            loss_net = criterion_net(net_representation.flatten(), data['net'].y.to(device))
            loss = loss_node + 0.001*loss_net
            loss.backward()
            optimizer.step()
            print(loss)
            print("debug finished")
            raise()

total_train_acc = 0
total_train_net_l1 = 0
train_acc, train_net_l1 = 0, 0
all_train_idx = 0
for data in tqdm([h_dataset[idx] for idx in all_train_indices]):
    try:
        node_representation, net_representation = model(data, device)
        train_acc = compute_accuracy(node_representation, data['node'].y.to(device))
        train_net_l1 = torch.nn.functional.l1_loss(net_representation.flatten(), data['net'].y.to(device)).item()
    except:
        print("OOM")
        continue
    
    total_train_acc += train_acc
    total_train_net_l1 += train_net_l1
    all_train_idx += 1

total_val_acc = 0
total_val_net_l1 = 0
val_acc, val_net_l1 = 0, 0
all_valid_idx = 0
for data in tqdm([h_dataset[idx] for idx in all_valid_indices]):
    try:
        node_representation, net_representation = model(data, device)
        val_acc = compute_accuracy(node_representation, data['node'].y.to(device))
        val_net_l1 = torch.nn.functional.l1_loss(net_representation.flatten(), data['net'].y.to(device)).item()
    except:
        print("OOM")
        continue
    
    total_val_acc += val_acc
    total_val_net_l1 += val_net_l1
    all_valid_idx += 1

total_test_acc = 0
total_test_net_l1 = 0
test_acc, test_net_l1 = 0, 0
all_test_idx = 0
for data in tqdm([h_dataset[idx] for idx in all_test_indices]):
    try:
        node_representation, net_representation = model(data, device)
        test_acc = compute_accuracy(node_representation, data['node'].y.to(device))
        test_net_l1 = torch.nn.functional.l1_loss(net_representation.flatten(), data['net'].y.to(device)).item()
    except:
        print("OOM")
        continue
    
    total_test_acc += test_acc
    total_test_net_l1 += test_net_l1
    all_test_idx += 1

np.save("all_eval_metric.npy", [total_train_acc/all_train_idx, total_train_net_l1/all_train_idx, total_val_acc/all_valid_idx, total_val_net_l1/all_valid_idx, total_test_acc/all_test_idx, total_test_net_l1/all_test_idx])
