import os
import numpy as np
import pickle
import wandb

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, HeteroData

from torch_geometric.utils import scatter
from tqdm import tqdm
from collections import Counter

from utils import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
sys.path.insert(1, 'data/')
from pyg_dataset import NetlistDataset
from gen_h_dataset import process_vlsi_dataset

sys.path.append("models/")
sys.path.append("models/layers/")
from models.model_unet import GNN_node
test = False # if only test but not train
restart = False # if restart training
reload_dataset = True # if reload already processed h_dataset

model_type = "dehnn"
num_layer = 2
num_dim = 32
vn = True
cv = True
aggr = "add"
device = "cuda"

learning_rate = 0.0005

target_data_dir = "data/target_data"
load_indices = np.array(['221', '181', '226', '206', '191', '190', '192', '182', '222', '197', '71', '81', '151', '161', '106', '160', '112', '75', '37', '82', '21', '45', '102', '140', '7'])

print("Loading the design with indices: ", load_indices)

if not reload_dataset:
    dataset = NetlistDataset(data_dir="data/all_designs_netlist_data", load_pe = True, pl = True, processed = True, load_indices = load_indices)
    h_dataset = process_vlsi_dataset(dataset, target_data_dir)

else:
    h_dataset = torch.load("h_dataset.pt")
    
sys.path.append("models/layers/")

dataset = None

h_data = h_dataset[0]

if restart:
    model = torch.load(f"{model_type}_{num_layer}_{num_dim}_{vn}_{cv}_unet.pt")
else:
    model = GNN_node(num_layer, num_dim, 1, 1, node_dim = h_data['node'].node_features.shape[1] + 2, net_dim = h_data['net'].net_features.shape[1] + 2, gnn_type=model_type, vn=vn, cv=cv, aggr=aggr, JK="Normal").to(device)

print(model)
    
criterion_node = nn.MSELoss()
criterion_vn_node = nn.MSELoss()
criterion_net = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
total_num_placements = 25*len(h_dataset)
load_data_indices = [idx for idx in range(total_num_placements)]
all_train_indices, all_valid_indices, all_test_indices = load_data_indices[:20*25], load_data_indices[20*25:], load_data_indices[20*25:]
wandb.init(project="de_hnn_tx", config={"lr": learning_rate, "architecture": model_type, "num_layer": num_layer, "num_dim": num_dim, "aggr": aggr, "vn": vn, "cv": cv})

best_total_val = None
for epoch in range(100):
    np.random.shuffle(all_train_indices)
    loss_node_all = 0
    loss_vn_all = 0
    loss_net_all = 0

    model.train()
    all_train_idx = 0
    for data_idx in tqdm(all_train_indices):
        data = h_dataset[data_idx//25]
        pos_lst, pos_lst_net, edge_attr, target_node, target_net = data.variant_data_lst[data_idx%25]
        optimizer.zero_grad()
        data['node'].x = torch.concat([data['node'].node_features, pos_lst], dim=1)
        data['net'].x = torch.concat([data['net'].net_features, pos_lst_net], dim=1)
        data['node', 'as_a_sink_of', 'net'].edge_attr = edge_attr
        data.pos_lst = pos_lst
        data.num_sites_x = 206
        data.num_sites_y = 300

        node_representation, net_representation, vn_representation = model(data, device)
        target_map = convert_to_util_map(pos_lst, target_node.to(device), 206, 300, device, "amax").flatten()
        
        loss_node = torch.tensor([1.0]) #criterion_node(node_representation.flatten(), target_node.to(device))
        loss_net = torch.tensor([1.0]) #criterion_net(net_representation.flatten(), target_net.to(device))
        loss_vn = criterion_vn_node(vn_representation.flatten(), target_map.to(device))
        
        loss = loss_vn 

        loss.backward()
        optimizer.step()
        loss_node_all += loss_node.item()
        loss_vn_all += loss_vn.item()
        loss_net_all += loss_net.item()
        all_train_idx += 1

    wandb.log({
        "loss_vn": loss_vn_all/all_train_idx
    })

    model.eval()
    with torch.no_grad():
        loss_node_all = 0
        loss_vn_all = 0
        loss_net_all = 0
        all_train_idx = 0
        for data_idx in tqdm(all_valid_indices):
            data = h_dataset[data_idx//25]
            pos_lst, pos_lst_net, edge_attr, target_node, target_net = data.variant_data_lst[data_idx%25]
            data['node'].x = torch.concat([data['node'].node_features, pos_lst], dim=1)
            data['net'].x = torch.concat([data['net'].net_features, pos_lst_net], dim=1)
            data['node', 'as_a_sink_of', 'net'].edge_attr = edge_attr
            data.pos_lst = pos_lst
            data.num_sites_x = 206
            data.num_sites_y = 300

            node_representation, net_representation, vn_representation = model(data, device)
            target_map = convert_to_util_map(pos_lst, target_node.to(device), 206, 300, device, "amax").flatten()

            loss_node = torch.tensor([1.0])#criterion_node(target_node.flatten(), target_node.to(device))
            loss_net = torch.tensor([1.0])#criterion_net(target_net.flatten(), target_net.to(device))
            loss_vn = criterion_vn_node(vn_representation.flatten(), target_map.to(device))

            loss_node_all += loss_node.item()
            loss_vn_all += loss_vn.item()
            loss_net_all += loss_net.item()

            all_train_idx += 1


    wandb.log({
        "val_loss_vn": loss_vn_all/all_train_idx,
    })
            
    if (best_total_val is None) or ((loss_vn_all/all_train_idx) < best_total_val):
        best_total_val = loss_vn_all/all_train_idx
        torch.save(model, f"{model_type}_{num_layer}_{num_dim}_{vn}_{cv}_unet.pt")
