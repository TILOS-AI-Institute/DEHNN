import os
import numpy as np
import pickle

import torch
import torch.nn as nn
import torch.optim as optim

from torch_geometric.data import Dataset
from torch_geometric.data import Data, HeteroData
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool

import time
import wandb
from tqdm import tqdm
from collections import Counter

from utils import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
sys.path.insert(1, 'data/')
from pyg_dataset import NetlistDataset

sys.path.append("models/layers/")
from models.model_att import GNN_node
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

def compute_r_squared(y_true, y_pred):
    """
    Compute R-squared for a regression model.
    
    Parameters:
        y_true (torch.Tensor): Ground truth (actual values).
        y_pred (torch.Tensor): Model predictions.

    Returns:
        float: R-squared value.
    """
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    y_mean = torch.mean(y_true)
    
    # Total Sum of Squares (TSS)
    tss = torch.sum((y_true - y_mean) ** 2)
    # Residual Sum of Squares (RSS)
    rss = torch.sum((y_true - y_pred) ** 2)
    # Compute R-squared
    r_squared = 1 - (rss / tss)
    return r_squared.item()

def weighted_mse_loss(input, target, weight):
    return torch.mean(weight * (input - target) ** 2)

#hyperparameter
test = False # if only test but not train
restart = False # if restart training
reload_dataset = True # if reload already processed h_dataset

model_type = "dehnn"
num_layer = 6
num_dim = 48
vn = False
trans = False
aggr = "add"
device = "cuda"

learning_rate = 0.001

target_data_dir = "data/target_data"
load_indices = np.array(['105', '120', '231', '151', '55', '192', '2', '41', '81', '92', '130', '150', '147', '16', '106', '116', '132', '160', '61', '10', '166', '220', '126', '215', '71'])
load_indices =  np.array(['142'])

print("Loading the design with indices: ", load_indices)

if not reload_dataset:
    dataset = NetlistDataset(data_dir="data/all_designs_netlist_data", load_pe = True, pl = True, processed = True, load_indices = load_indices)
    h_dataset = []
    for data in tqdm(dataset):    
        #filter out all nets with high fanout
        out_degrees = data.net_features[:, 1]
        mask = (out_degrees < 3000)
        mask_edges = mask[data.edge_index_source_to_net[1]] 
        filtered_edge_index_source_to_net = data.edge_index_source_to_net[:, mask_edges]
        data.edge_index_source_to_net = filtered_edge_index_source_to_net

        mask_edges = mask[data.edge_index_sink_to_net[1]] 
        filtered_edge_index_sink_to_net = data.edge_index_sink_to_net[:, mask_edges]
        data.edge_index_sink_to_net = filtered_edge_index_sink_to_net
        
        h_data = HeteroData()
        h_data['node'].node_features = data.node_features
        h_data['net'].x = data.net_features
        h_data.num_instances = data.node_features.shape[0]
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
                
                node_congestion = node_congestion/580 #for normalization, where 580 is a baseline  
                #local_to_vn_edge_index = torch.vstack([torch.tensor([idx for idx in range(h_data['node'].node_features.shape[0])]), batch])
                variant_data_lst.append((pos_lst, edge_attr, node_congestion, net_hpwl, batch, num_vn)) 
    
        h_data['variant_data_lst'] = variant_data_lst
    
        #print(design_num, len(variant_data_lst))
        
        h_dataset.append(h_data)

        torch.save(h_dataset, "h_dataset.pt")

else:
    h_dataset = torch.load("h_dataset.pt")
    
sys.path.append("models/layers/")

dataset = None

h_data = h_dataset[0]

if restart:
    model = torch.load(f"{model_type}_{num_layer}_{num_dim}_{vn}_{trans}_model.pt")
else:
    model = GNN_node(num_layer, num_dim, 1, 1, node_dim = h_data['node'].node_features.shape[1] + 2, net_dim = h_data['net'].x.shape[1], gnn_type=model_type, vn=vn, trans=trans, aggr=aggr, JK="Normal").to(device)

print(model)
    
criterion_node = nn.MSELoss()
criterion_net = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
load_data_indices = [idx for idx in range(len(h_dataset))]
all_train_indices, all_valid_indices, all_test_indices = load_data_indices[:20], load_data_indices[20:22], load_data_indices[22:]
best_total_val = None

wandb.init(project="netlist_hyper", config={"lr": learning_rate, "architecture": model_type, "num_layer": num_layer, "num_dim": num_dim, "aggr": aggr, "vn": vn, "trans": trans})
#for weighted MSE
#y_max_all = []
#for data_idx in load_data_indices:
#    data = h_dataset[data_idx]
#    y_max_each = []
#    for inner_data_idx in range(len(data.variant_data_lst)):
#        pos_lst, edge_attr, node_congestion, net_hpwl, batch, num_vn = data.variant_data_lst[inner_data_idx]
#        y_weight_each = node_congestion/torch.max(node_congestion)
#        top_mask = ((node_congestion >= np.percentile(node_congestion, 90)).float() * 2) + 1
#        y_weight_each = y_weight_each * top_mask
#        y_max_each.append(y_weight_each)
#    y_max_all.append(y_max_each)


for epoch in range(300):
    np.random.shuffle(all_train_indices)
    loss_node_all = 0
    loss_net_all = 0
    val_loss_node_all = 0
    val_loss_net_all = 0
    loss_node_rsquare_all = 0
    loss_net_rsquare_all = 0
    #precision_all = 0
    #recall_all = 0
    
    all_train_idx = 0
    for data_idx in tqdm(all_train_indices):
        data = h_dataset[data_idx]
        #y_max_each = y_max_all[data_idx]
        
        for inner_data_idx in tqdm(range(len(data.variant_data_lst))):
            pos_lst, edge_attr, target_node, target_net, batch, num_vn = data.variant_data_lst[inner_data_idx]
            optimizer.zero_grad()
            
            data['node'].x = torch.concat([data['node'].node_features, pos_lst], dim=1)
            data.batch = batch
            data.num_vn = num_vn
            data['node', 'as_a_sink_of', 'net'].edge_attr = edge_attr
            data.vn = torch.concat([global_add_pool(data['node'].node_features, batch), 
                        global_add_pool(pos_lst, batch), 
                        global_max_pool(data['node'].node_features, batch), 
                        global_max_pool(pos_lst, batch)], dim=1)
            #data.local_to_vn_edge_index = local_to_vn_edge_index
            node_representation, net_representation = model(data, device)

            #y_weight_each = y_max_each[inner_data_idx]
             #loss_node = weighted_mse_loss(node_representation.flatten(), target_node.to(device), y_weight_each.to(device))
            
            loss_node = criterion_node(node_representation.flatten(), target_node.to(device))
            loss_net = criterion_net(net_representation.flatten(), target_net.to(device))
            loss = loss_node + 0.001*loss_net
            loss.backward()
            optimizer.step()
            
            loss_node_all += loss_node.item()
            loss_net_all += loss_net.item()

            loss_node_rsquare = compute_r_squared(node_representation.flatten(), target_node.to(device))
            loss_net_rsquare = compute_r_squared(net_representation.flatten(), target_net.to(device))
            loss_node_rsquare_all += loss_node_rsquare
            loss_net_rsquare_all += loss_net_rsquare
            
            all_train_idx += 1

    print(loss_node_all/all_train_idx, loss_net_all/all_train_idx, loss_node_rsquare/all_train_idx, loss_net_rsquare_all/all_train_idx)
    wandb.log({
        "loss_node": loss_node_all/all_train_idx,
        "loss_net": loss_net_all/all_train_idx,
        "loss_node_rsquare": loss_node_rsquare/all_train_idx, 
        "loss_net_rsquare": loss_net_rsquare_all/all_train_idx
    })

    # all_valid_idx = 0
    # for data_idx in tqdm(all_valid_indices):
    #     data = h_dataset[data_idx]
    #     #y_max_each = y_max_all[data_idx]

    #     for inner_data_idx in tqdm(range(len(data.variant_data_lst))):
    #         pos_lst, edge_attr, target_node, target_net, batch, num_vn = data.variant_data_lst[inner_data_idx]
            
    #         data['node'].x = torch.concat([data['node'].node_features, pos_lst], dim=1)
    #         data.batch = batch
    #         data.num_vn = num_vn
    #         data['node', 'as_a_sink_of', 'net'].edge_attr = edge_attr
    #         data.vn = torch.concat([global_add_pool(data['node'].node_features, batch), 
    #                                 global_add_pool(pos_lst, batch), 
    #                                 global_max_pool(data['node'].node_features, batch), 
    #                                 global_max_pool(pos_lst, batch)], dim=1)
    #         node_representation, net_representation = model(data, device)

    #         #y_weight_each = y_max_each[inner_data_idx]
    #         val_loss_node = criterion_node(node_representation.flatten(), target_node.to(device))
    #         #val_loss_node = weighted_mse_loss(node_representation.flatten(), target_node.to(device), y_weight_each.to(device))
    #         val_loss_net = criterion_net(net_representation.flatten(), target_net.to(device))
    #         val_loss_node_all +=  val_loss_node.item()
    #         val_loss_net_all += val_loss_net.item()
    #         all_valid_idx += 1

    # print(val_loss_node_all/all_valid_idx, val_loss_net_all/all_valid_idx)
    # # wandb.log({
    # #     "val_loss_node": val_loss_node_all/all_valid_idx,
    # #     "val_loss_net": val_loss_net_all/all_valid_idx,
    # # })
            
    # if (epoch // 10 > 0) and (epoch % 10 == 0):
    #     if (best_total_val is None) or ((loss_node_all/all_train_idx) < best_total_val):
    #         best_total_val = loss_node_all/all_train_idx
    #         torch.save(model, f"{model_type}_{num_layer}_{num_dim}_{vn}_{trans}_model.pt")
