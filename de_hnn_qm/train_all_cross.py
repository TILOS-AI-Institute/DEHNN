import os
import numpy as np
import pickle
import torch
import torch.nn
import time
from torch_geometric.data import Dataset
from torch_geometric.data import Data, HeteroData
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from utils import *
import torch.nn as nn
import torch.optim as optim
import wandb
from tqdm import tqdm

from collections import Counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
sys.path.insert(1, 'data/')

from pyg_dataset import NetlistDataset

sys.path.append("models/layers/")
from models.model import GNN_node

from sklearn.metrics import accuracy_score, precision_score, recall_score

# Function to compute accuracy, precision, and recall
def compute_metrics(true_labels, predicted_labels):
    # Accuracy
    accuracy = accuracy_score(true_labels, predicted_labels)
    
    # Precision
    precision = precision_score(true_labels, predicted_labels, average='binary')
    
    # Recall
    recall = recall_score(true_labels, predicted_labels, average='binary')
    
    return accuracy, precision, recall

test = False
restart = False
debug_mode = False
reload_dataset = True
model_type = "dehnn"

if debug_mode:
    load_indices = [0, 1]
else:
    load_indices = None

if not debug_mode:
    device = "cuda"
    #wandb.init(project="netlist_cross_designs", config={"lr": 0.001})
else:
    device = "cpu"

if not reload_dataset:
    dataset = NetlistDataset(data_dir="data/superblue", load_pe = True, pl = True, processed = True, load_indices=load_indices)
    h_dataset = []
    for data in tqdm(dataset):
        print(data)
        num_instances = data.node_congestion.shape[0]
        data.num_instances = num_instances
        data.edge_index_sink_to_net[1] = data.edge_index_sink_to_net[1] - num_instances
        data.edge_index_source_to_net[1] = data.edge_index_source_to_net[1] - num_instances
        out_degrees = data.net_features[:, 0]
        mask = (out_degrees < 3000)
        mask_edges = mask[data.edge_index_source_to_net[1]] 
        filtered_edge_index = data.edge_index_source_to_net[:, mask_edges]
        data.edge_index_source_to_net = filtered_edge_index
    
        mask_edges = mask[data.edge_index_sink_to_net[1]] 
        filtered_edge_index = data.edge_index_sink_to_net[:, mask_edges]
        
        h_data = HeteroData()
        h_data['node'].x = data.node_features
        h_data['node'].y = data.node_congestion
        
        h_data['net'].x = data.net_features
        h_data['net'].y = data.net_hpwl
        
        h_data['node', 'as_a_sink_of', 'net'].edge_index, h_data['node', 'as_a_sink_of', 'net'].edge_weight = gcn_norm(data.edge_index_sink_to_net, add_self_loops=False)
        h_data['node', 'as_a_source_of', 'net'].edge_index = data.edge_index_source_to_net
        h_data['node', 'as_a_sink_of', 'net'].edge_attr = None
        
        h_data.batch = data.batch
        h_data.num_vn = data.num_vn
        h_data.num_instances = num_instances
        _, h_data['net', 'sink_to', 'node'].edge_weight = gcn_norm(torch.flip(h_data['node', 'as_a_sink_of', 'net'].edge_index, dims=[0]), add_self_loops=False)

        h_data['design_name'] = data['design_name']
        h_dataset.append(h_data)
        
    torch.save(h_dataset, "h_dataset.pt")
elif not debug_mode:
    dataset = torch.load("h_dataset.pt")
    h_dataset = []
    for data in dataset:
        h_dataset.append(data)
    
sys.path.append("models/layers/")

h_data = h_dataset[0]
#all_train_indices, all_valid_indices, all_test_indices = pickle.load(open("cross_design_data_split.pt", "rb"))
all_indices = [idx for idx in range(len(h_dataset))]
for train_idx in range(len(all_indices)):
    if restart:
        model = torch.load(f"best_{model_type}_model_single.pt")
    else:
        model = GNN_node(3, 32, 4, 1, node_dim = h_data['node'].x.shape[1], net_dim = h_data['net'].x.shape[1], gnn_type=model_type, vn=True, JK="Normal").to(device)
    criterion_node = nn.MSELoss()
    criterion_net = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    best_total_val = None
    all_train_indices, all_valid_indices, all_test_indices = all_indices[train_idx:train_idx+1], all_indices[train_idx:train_idx+1], all_indices[train_idx:train_idx+1]
    design_name = h_dataset[all_train_indices[0]]['design_name']

    if not debug_mode:
        for epoch in range(500):
            np.random.shuffle(all_train_indices)
            loss_node_all = 0
            loss_net_all = 0
            val_loss_node_all = 0
            val_loss_net_all = 0

            precision_all = 0
            recall_all = 0
            
            model.train()
            all_train_idx = 0
            for data_idx in tqdm(all_train_indices):
                data = h_dataset[data_idx]
                optimizer.zero_grad()
                node_representation, net_representation = model(data, device)
                
                if model_type != "gat":
                    target_node = data['node'].y.float()
                    target_net = data['net'].y.float()
                else:
                    target_node = data.node_congestion
                    target_net = data.net_hpwl
                
                loss_node = criterion_node(node_representation, target_node.to(device))
                loss_net = criterion_net(net_representation.flatten(), target_net.to(device))
                loss = loss_node + 0.001*loss_net
                loss.backward()
                optimizer.step()
                
                loss_node_all += loss_node.item()
                loss_net_all += loss_net.item()
                all_train_idx += 1
            print(loss_node_all/all_train_idx, loss_net_all/all_train_idx)
        
            model.eval()
            all_valid_idx = 0
            for data_idx in tqdm(all_valid_indices):
                #try:
                data = h_dataset[data_idx]
                node_representation, net_representation = model(data, device)
                
                if model_type != "gat":
                    target_node = data['node'].y.float()
                    target_net = data['net'].y.float()
                else:
                    target_node = data.node_congestion
                    target_net = data.net_hpwl
                    
                val_loss_node = criterion_node(node_representation, target_node.to(device))
                val_loss_net = criterion_net(net_representation.flatten(), target_net.to(device))
                val_loss_node_all +=  val_loss_node.item()
                val_loss_net_all += val_loss_net.item()
                all_valid_idx += 1
                # except:
                #     print("OOM")
            print(val_loss_node_all/all_valid_idx, val_loss_net_all/all_valid_idx)

            if (epoch // 10 > 0) and (epoch % 10 == 0):
                #torch.save(model, f"{model_type}_model.pt")
                    
                if (best_total_val is None) or ((val_loss_node_all/all_valid_idx) < best_total_val):
                    best_total_val = val_loss_node_all/all_valid_idx
                    torch.save(model, f"best_{model_type}_model_{design_name}.pt")

# total_train_acc = 0
# total_train_net_l1 = 0
# train_l1, train_net_l1 = 0, 0
# all_train_idx = 0
# for data in tqdm([h_dataset[idx] for idx in all_train_indices]):
#     try:
#         node_representation, net_representation = model(data, device)
#         node_representation = torch.clamp(node_representation, min=0, max=8)
#         #node_representation = torch.clamp(node_representation, min=0, max=90)
#         #train_acc = compute_accuracy(node_representation, data['node'].y.to(device))
#         train_l1 = torch.nn.functional.l1_loss(node_representation.flatten(), data['node'].y.to(device)).item()
#         train_net_l1 = torch.nn.functional.l1_loss(net_representation.flatten(), data['net'].y.to(device)).item()
#     except:
#         print("OOM")
#         continue
    
#     total_train_acc += train_l1
#     total_train_net_l1 += train_net_l1
#     all_train_idx += 1

# total_val_acc = 0
# total_val_net_l1 = 0
# val_l1, val_net_l1 = 0, 0
# all_valid_idx = 0
# for data in tqdm([h_dataset[idx] for idx in all_valid_indices]):
#     try:
#         node_representation, net_representation = model(data, device)
#         node_representation = torch.clamp(node_representation, min=0, max=8)
#         #node_representation = torch.clamp(node_representation, min=0, max=90)
#         #val_acc = compute_accuracy(node_representation, data['node'].y.to(device))
#         val_l1 = torch.nn.functional.l1_loss(node_representation.flatten(), data['node'].y.to(device)).item()
#         val_net_l1 = torch.nn.functional.l1_loss(net_representation.flatten(), data['net'].y.to(device)).item()
#     except:
#         print("OOM")
#         continue
    
#     total_val_acc += val_l1
#     total_val_net_l1 += val_net_l1
#     all_valid_idx += 1

# total_test_acc = 0
# total_test_net_l1 = 0
# test_l1, test_net_l1 = 0, 0
# all_test_idx = 0
# for data in tqdm([h_dataset[idx] for idx in all_test_indices]):
#     try:
#         node_representation, net_representation = model(data, device)
#         node_representation = torch.clamp(node_representation, min=0, max=8)
#         #node_representation = torch.clamp(node_representation, min=0, max=90)
#         #test_acc = compute_accuracy(node_representation, data['node'].y.to(device))
#         test_l1 = torch.nn.functional.l1_loss(node_representation.flatten(), data['node'].y.to(device)).item()
#         test_net_l1 = torch.nn.functional.l1_loss(net_representation.flatten(), data['net'].y.to(device)).item()
#     except:
#         print("OOM")
#         continue
    
#     total_test_acc += test_l1
#     total_test_net_l1 += test_net_l1
#     all_test_idx += 1

# np.save("all_eval_metric.npy", [total_train_acc/all_train_idx, total_train_net_l1/all_train_idx, total_val_acc/all_valid_idx, total_val_net_l1/all_valid_idx, total_test_acc/all_test_idx, total_test_net_l1/all_test_idx])
