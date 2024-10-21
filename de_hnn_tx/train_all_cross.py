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

def weighted_mse_loss(input, target, weight):
    return torch.mean(weight * (input - target) ** 2)

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
else:
    device = "cpu"

target_data_dir = "data/target_data"

load_indices = np.load("data/all_target_design_nums.npy")

load_indices = load_indices[:12]

print(load_indices)

dataset = NetlistDataset(data_dir="data/all_designs_netlist_data", load_pe = True, pl = True, processed = True, load_indices = load_indices)

if not reload_dataset:
    h_dataset = []
    for data in tqdm(dataset):    
        out_degrees = data.net_features[:, 1]
        mask = (out_degrees < 3000)
        mask_edges = mask[data.edge_index_source_to_net[1]] 
        filtered_edge_index = data.edge_index_source_to_net[:, mask_edges]
        data.edge_index_source_to_net = filtered_edge_index
    
        mask_edges = mask[data.edge_index_sink_to_net[1]] 
        filtered_edge_index = data.edge_index_sink_to_net[:, mask_edges]
        data.edge_index_sink_to_net = filtered_edge_index
        
        h_data = HeteroData()
        h_data['node'].node_features = data.node_features
        h_data['net'].x = data.net_features
        h_data.num_instances = len(data.node_features)
        h_data['node', 'as_a_sink_of', 'net'].edge_index, h_data['node', 'as_a_sink_of', 'net'].edge_weight = gcn_norm(data.edge_index_sink_to_net, add_self_loops=False)
        h_data['node', 'as_a_source_of', 'net'].edge_index = data.edge_index_source_to_net
        _, h_data['net', 'sink_to', 'node'].edge_weight = gcn_norm(torch.flip(h_data['node', 'as_a_sink_of', 'net'].edge_index, dims=[0]), add_self_loops=False)
        
        design_num = data['design_name'].split("_")[1]
        variant_data_lst = []
        for design_fp in os.listdir(target_data_dir):
            design_num_to_match = design_fp.split("_")[1]
            if design_num == design_num_to_match:
                with open(os.path.join(target_data_dir, design_fp, 'node_loc_x.pkl'), 'rb') as f:
                    node_loc_x = pickle.load(f)
    
                with open(os.path.join(target_data_dir, design_fp, 'node_loc_y.pkl'), 'rb') as f:
                    node_loc_y = pickle.load(f)
    
                with open(os.path.join(target_data_dir, design_fp, 'target_net_hpwl.pkl'), 'rb') as f:
                    net_hpwl = pickle.load(f)
                
                with open(os.path.join(target_data_dir, design_fp, 'target_node_utilization.pkl'), 'rb') as f:
                    node_congestion = pickle.load(f)
    
                pos_lst = torch.tensor(np.vstack([node_loc_x, node_loc_y]).T)
    
                assert pos_lst.shape[0] == h_data['node'].node_features.shape[0]
                
                node_congestion = torch.tensor(node_congestion).float()
                net_hpwl = torch.tensor(net_hpwl).float()
    
                node_pos_lst = pos_lst[data.edge_index_sink_to_net[0]]
                net_pos_lst = pos_lst[data.edge_index_source_to_net[0]][data.edge_index_sink_to_net[1]]
                edge_attr = torch.sum(torch.abs(node_pos_lst - net_pos_lst), dim=1)
    
                file_name = os.path.join(target_data_dir, design_fp, 'pl_part_dict.pkl')
                f = open(file_name, 'rb')
                part_dict = pickle.load(f)
                f.close()
                batch = [part_dict[idx] for idx in range(node_congestion.shape[0])]
                num_vn = len(np.unique(batch))
                batch = torch.tensor(batch).long()
                variant_data_lst.append((pos_lst, edge_attr, node_congestion, net_hpwl, batch, num_vn))
    
        h_data['variant_data_lst'] = variant_data_lst
    
        print(design_num, len(variant_data_lst))
        
        h_dataset.append(h_data)

        torch.save(h_dataset, "h_dataset.pt")

else:
    h_dataset = torch.load("h_dataset.pt")
    
sys.path.append("models/layers/")

dataset = None

h_data = h_dataset[0]

if restart:
    model = torch.load(f"best_{model_type}_model.pt")
else:
    model = GNN_node(4, 32, 1, 1, node_dim = h_data['node'].node_features.shape[1] + 2, net_dim = h_data['net'].x.shape[1], gnn_type=model_type, vn=True, JK="Normal").to(device)

print(model)
    
criterion_node = nn.MSELoss()
criterion_net = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
load_data_indices = [idx for idx in range(len(h_dataset))]
all_train_indices, all_valid_indices, all_test_indices = load_data_indices[:1], load_data_indices[:1], load_data_indices[:1] #pickle.load(open("cross_design_data_split.pt", "rb"))
best_total_val = None

# for weighted MSE
y_max_all = []
for data_idx in load_data_indices:
    data = h_dataset[data_idx]
    y_max_each = []
    for inner_data_idx in range(len(data.variant_data_lst)):
        pos_lst, edge_attr, node_congestion, net_hpwl, batch, num_vn = data.variant_data_lst[inner_data_idx]
        y_weight_each = node_congestion/torch.max(node_congestion)
        top_mask = (node_congestion >= np.percentile(node_congestion, 90))
        top_mask = (top_mask.float() * 2) + 1
        y_weight_each = y_weight_each * top_mask
        y_max_each.append(y_weight_each)
    y_max_all.append(y_max_each)


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
    for data_idx in tqdm(all_train_indices):
        data = h_dataset[data_idx]
        y_max_each = y_max_all[data_idx]

        for inner_data_idx in tqdm(range(len(data.variant_data_lst))):
            pos_lst, edge_attr, node_congestion, net_hpwl, batch, num_vn = data.variant_data_lst[inner_data_idx]
            optimizer.zero_grad()
            data['node'].x = torch.concat([data['node'].node_features, pos_lst + (torch.rand(1) * 10)], dim=1)
            data.batch = batch
            data.num_vn = num_vn
            data['node', 'as_a_sink_of', 'net'].edge_attr = edge_attr
            node_representation, net_representation = model(data, device)
            target_node = node_congestion    
            target_net = net_hpwl

            y_weight_each = y_max_each[inner_data_idx]
            #loss_node = criterion_node(node_representation.flatten(), target_node.to(device))
            loss_node = weighted_mse_loss(node_representation.flatten(), target_node.to(device), y_weight_each.to(device))
            loss_net = criterion_net(net_representation.flatten(), target_net.to(device))
            loss = 100*loss_node + 0.001*loss_net
            loss.backward()
            optimizer.step()
            
            loss_node_all += loss_node.item()
            loss_net_all += loss_net.item()
            all_train_idx += 1

    print(loss_node_all/all_train_idx, loss_net_all/all_train_idx)
    
    # optimizer.zero_grad()
    # all_valid_idx = 0
    # for data_idx in tqdm(all_valid_indices):
    #     data = h_dataset[data_idx]
    #     y_max_each = y_max_all[data_idx]

    #     for inner_data_idx in tqdm(range(len(data.variant_data_lst))):
    #         pos_lst, edge_attr, node_congestion, net_hpwl, batch, num_vn = data.variant_data_lst[inner_data_idx]
    #         data['node'].x = torch.concat([data['node'].node_features, pos_lst + (torch.rand(1) * 10)], dim=1)
    #         data.batch = batch
    #         data.num_vn = num_vn
    #         data['node', 'as_a_sink_of', 'net'].edge_attr = edge_attr
    #         node_representation, net_representation = model(data, device)
    #         target_node = node_congestion    
    #         target_net = net_hpwl

    #         y_weight_each = y_max_each[inner_data_idx]
            
    #         #val_loss_node = criterion_node(node_representation.flatten(), target_node.to(device))
    #         val_loss_node = weighted_mse_loss(node_representation.flatten(), target_node.to(device), y_weight_each.to(device))
    #         val_loss_net = criterion_net(net_representation.flatten(), target_net.to(device))
    #         val_loss_node_all +=  val_loss_node.item()
    #         val_loss_net_all += val_loss_net.item()
    #         all_valid_idx += 1

    # print(val_loss_node_all/all_valid_idx, val_loss_net_all/all_valid_idx)

    if (epoch // 10 > 0) and (epoch % 10 == 0):
        torch.save(model, f"{model_type}_model_no_trans.pt")
            
        # if (best_total_val is None) or ((val_loss_node_all/all_valid_idx) < best_total_val):
        #     best_total_val = val_loss_node_all/all_valid_idx
        #     torch.save(model, f"best_{model_type}_model.pt")

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
