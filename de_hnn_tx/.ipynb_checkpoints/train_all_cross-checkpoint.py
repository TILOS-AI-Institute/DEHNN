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
    dataset = NetlistDataset(data_dir="data/cross_design_data", load_pe = True, pl = True, load_indices=load_indices)
    h_dataset = []
    for data in tqdm(dataset):
        data.node_congestion = (data.node_congestion >= 3).long()
        num_instances = data.node_congestion.shape[0]
        data.num_instances = num_instances
        pos_lst = data.pos
        
        node_pos_lst = pos_lst[data.edge_index_sink_to_net[0]]
        net_pos_lst = pos_lst[data.edge_index_source_to_net[0]][data.edge_index_sink_to_net[1]]
        
        out_degrees = data.net_features[:, 1]
        mask = (out_degrees < 3000)
        mask_edges = mask[data.edge_index_source_to_net[1]] 
        filtered_edge_index = data.edge_index_source_to_net[:, mask_edges]
        data.edge_index_source_to_net = filtered_edge_index
        
        mask_edges = mask[data.edge_index_sink_to_net[1]]
        filtered_edge_index = data.edge_index_sink_to_net[:, mask_edges]
        data.edge_index_sink_to_net = filtered_edge_index
        
        h_data = HeteroData()
        h_data['node'].x = data.node_features
        h_data['node'].y = data.node_congestion
        
        h_data['net'].x = data.net_features
        h_data['net'].y = data.net_hpwl
        
        h_data['node', 'as_a_sink_of', 'net'].edge_index, h_data['node', 'as_a_sink_of', 'net'].edge_weight = gcn_norm(data.edge_index_sink_to_net, add_self_loops=False)
        h_data['node', 'as_a_source_of', 'net'].edge_index = data.edge_index_source_to_net
        h_data['node', 'as_a_sink_of', 'net'].edge_attr = torch.sum(torch.abs(node_pos_lst - net_pos_lst), dim=1)[mask_edges]
        
        h_data.batch = data.batch
        h_data.num_vn = data.num_vn
        h_data.num_instances = num_instances
        
        h_data['node', 'as_a_sink_of', 'net'].edge_index, h_data['node', 'as_a_sink_of', 'net'].edge_weight = gcn_norm(data.edge_index_sink_to_net, add_self_loops=False)
        h_data['node', 'as_a_source_of', 'net'].edge_index = data.edge_index_source_to_net
        _, h_data['net', 'sink_to', 'node'].edge_weight = gcn_norm(torch.flip(h_data['node', 'as_a_sink_of', 'net'].edge_index, dims=[0]), add_self_loops=False)

        h_data.batch = data.batch
        h_data.num_vn = data.num_vn
        h_data.num_instances = num_instances
        h_dataset.append(h_data)
        
    torch.save(h_dataset, "h_dataset.pt")
elif not debug_mode:
    h_dataset = torch.load("136_dataset.pt")
    for data in h_dataset:
        data['node'].x = torch.concat([data['node'].x, data.pos + (torch.rand(1) * 100)], dim=1)
    
sys.path.append("models/layers/")

h_data = h_dataset[0]

if restart:
    model = torch.load(f"best_{model_type}_model.pt")
else:
    if model_type == "gat":
        model = GNN_node(4, 32, 2, 1, node_dim = h_data.node_features.shape[1], net_dim = h_data.net_features.shape[1], gnn_type=model_type, vn=False).to(device)
    else:
        model = GNN_node(6, 32, 2, 1, node_dim = h_data['node'].x.shape[1], net_dim = h_data['net'].x.shape[1], gnn_type=model_type, vn=True, JK="Normal").to(device)
    
criterion_node = nn.CrossEntropyLoss()
criterion_net = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
all_train_indices, all_valid_indices, all_test_indices = pickle.load(open("data/cross_design_data_split.pt", "rb"))
best_total_val = None
#wandb.init(project="your_project_name", config={"lr": 0.001})
all_indices = [idx for idx in range(len(h_dataset))]
all_train_indices, all_valid_indices, all_test_indices = all_indices[:8], all_indices[8:], all_indices[8:]
#assert len(all_indices) >= 10

if not test:
    if not debug_mode:
        for epoch in range(1000):
            np.random.shuffle(all_train_indices)
            loss_node_all = 0
            loss_net_all = 0
            val_loss_node_all = 0
            val_loss_net_all = 0

            precision_all = 0
            recall_all = 0
            
            model.train()
            all_train_idx = 0
            for data in tqdm([h_dataset[idx] for idx in all_train_indices]):
                optimizer.zero_grad()
                data['node'].x = torch.concat([data['node'].x[:, :-2], data.pos + (torch.rand(1) * 100)], dim=1)
                node_representation, net_representation = model(data, device)
                
                if model_type != "gat":
                    target_node = data['node'].y.long()
                    target_net = data['net'].y.float()
                else:
                    target_node = data.node_congestion
                    target_net = data.net_hpwl
                    
                bias_distribution = torch.tensor([torch.mean(target_node.float()).item() * 10, 1.0])
                keep_probabilities = bias_distribution[target_node]
                random_values = torch.rand(len(target_node))
                keep_mask = random_values < keep_probabilities
                
                loss_node = criterion_node(node_representation[keep_mask], target_node[keep_mask].to(device))
                loss_net = criterion_net(net_representation.flatten(), target_net.to(device))
                loss = loss_node + 0.001*loss_net
                loss.backward()
                optimizer.step()
                
                loss_node_all += loss_node.item()
                loss_net_all += loss_net.item()
                all_train_idx += 1

                
            print(torch.mean(torch.argmax(node_representation, dim=1).float()))
            print(loss_node_all/all_train_idx, loss_net_all/all_train_idx)
            # wandb.log({
            #     "loss_node": loss_node_all/all_train_idx,
            #     "loss_net": loss_net_all/all_train_idx,
            # })
        
            model.eval()
            all_valid_idx = 0
            for data in tqdm([h_dataset[idx] for idx in all_valid_indices]):
                #try:
                node_representation, net_representation = model(data, device)
                
                if model_type != "gat":
                    target_node = data['node'].y.long()
                    target_net = data['net'].y.float()
                else:
                    target_node = data.node_congestion
                    target_net = data.net_hpwl
                    
                val_loss_node = criterion_node(node_representation, target_node.to(device))
                val_loss_net = criterion_net(net_representation.flatten(), target_net.to(device))
                val_loss_node_all +=  val_loss_node.item()
                val_loss_net_all += val_loss_net.item()
                all_valid_idx += 1
                #except:
                # print("OOM")
                # continue

            if (epoch // 10 > 0) and (epoch % 10 == 0):
                torch.save(model, f"{model_type}_model.pt")
                    
                if (best_total_val is None) or ((val_loss_node_all/all_valid_idx) < best_total_val):
                    best_total_val = val_loss_node_all/all_valid_idx
                    torch.save(model, f"best_{model_type}_model.pt")
    

        
            #print(val_loss_node_all/all_valid_idx, val_loss_net_all/all_valid_idx)
            # wandb.log({
            #     "val_loss_node": val_loss_node_all/all_valid_idx,
            #     "val_loss_net": val_loss_net_all/all_valid_idx,
            # })
    else:
        for data in tqdm(h_dataset):
            optimizer.zero_grad()
            #weights = data.weights.to(device)
            node_representation, net_representation = model(data, device)
            node_representation = torch.clamp(node_representation, min=0, max=8)
            loss_node = torch.mean(criterion_node(node_representation.flatten(), data['node'].y.float().to(device)*10))
            #loss_node = criterion_node(node_representation, data['node'].y.long().to(device))
            loss_net = criterion_net(net_representation.flatten(), data['net'].y.to(device))
            #loss = torch.mean(loss_node*weights) + 0.001*loss_net
            loss = loss_node + 0.001*loss_net
            loss.backward()
            optimizer.step()
            print(loss)
            print("debug finished")
            raise()

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
