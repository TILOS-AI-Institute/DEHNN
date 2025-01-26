import os
import numpy as np
import pickle
import wandb
import argparse

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
from models.model import GNN_node
from models.model_unet import GNN_node as GNN_node_unet

test = False  # if only test but not train
restart = False  # if restart training
reload_dataset = True  # if reload already processed h_dataset
model_type = "dehnn"
num_layer = 2
num_dim = 32
vn = True
cv = True
aggr = "add"
learning_rate = 0.0005
num_epoch = 500
init_patience = 50
patience = init_patience
unet_only = True  # True for UNet model, False for regular model
device = "cuda"
target_data_dir = "data/target_data"
load_indices = np.array(['221', '181', '226', '206', '191', '190', '192', '182', '222', '197', 
                       '71', '81', '151', '161', '106', '160', '112', '75', '37', '82', '21', '45', '102', '140', '7'])

if not reload_dataset:
    dataset = NetlistDataset(data_dir="data/all_designs_netlist_data", load_pe=True, pl=True, processed=True, load_indices=load_indices)
    h_dataset = process_vlsi_dataset(dataset, target_data_dir)
else:
    h_dataset = torch.load("h_dataset.pt")
        
print("Loading the design with indices: ", load_indices)

model_class = GNN_node_unet if unet_only else GNN_node
model_suffix = "unet" if unet_only else "model"
h_data = h_dataset[0]

if restart:
    model = torch.load(f"{model_type}_{num_layer}_{num_dim}_{vn}_{cv}_{model_suffix}.pt")
else:
    model = model_class(num_layer, num_dim, 1, 1, 
                      node_dim=h_data['node'].node_features.shape[1] + 2,
                      net_dim=h_data['net'].net_features.shape[1] + 2,
                      gnn_type=model_type, vn=vn, cv=cv,
                      aggr=aggr, JK="Normal").to(device)

print(model)
    
criterion_node = nn.MSELoss()
criterion_vn_node = nn.MSELoss()
criterion_net = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

total_num_placements = 25 * len(h_dataset)
load_data_indices = list(range(total_num_placements))
all_train_indices = load_data_indices[:20*25]
all_valid_indices = load_data_indices[20*25:]

wandb.init(project="de_hnn_tx", config={"lr": learning_rate, "architecture": model_type, "num_layer": num_layer, "num_dim": num_dim, "aggr": aggr, "vn": vn, "cv": cv})
best_total_val = None

for epoch in tqdm(range(num_epoch)):
    np.random.shuffle(all_train_indices)
    model.train()
    train_metrics = train_epoch(model, h_dataset, all_train_indices, optimizer, 
                              criterion_node, criterion_vn_node, criterion_net, 
                              device, unet_only)
    
    wandb.log({
        "loss_node": train_metrics["loss_node"],
        "loss_vn": train_metrics["loss_vn"],
        "loss_net": train_metrics["loss_net"],
    })

    model.eval()
    with torch.no_grad():
        val_metrics = validate_epoch(model, h_dataset, all_valid_indices,
                                  criterion_node, criterion_vn_node, criterion_net,
                                  device, unet_only)

    wandb.log({
        "val_loss_node": val_metrics["loss_node"],
        "val_loss_vn": val_metrics["loss_vn"],
        "val_loss_net": val_metrics["loss_net"],
        "patience": patience
    })

    if (best_total_val is None) or (val_metrics["loss_node"] < best_total_val):
        best_total_val = val_metrics["loss_node"]
        torch.save(model, f"{model_type}_{num_layer}_{num_dim}_{vn}_{cv}_{model_suffix}.pt")
        patience = init_patience
    else:
        patience -= 1

    if patience <= 0:
        break

