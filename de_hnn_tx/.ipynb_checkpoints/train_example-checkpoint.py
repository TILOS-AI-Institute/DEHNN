import os
import numpy as np
import pickle
import torch
import torch.nn
from torch_geometric.data import Dataset
from torch_geometric.data import Data
import torch.nn as nn
import torch.optim as optim

import sys
sys.path.insert(1, 'data/')

from pyg_dataset import NetlistDataset

sys.path.append("models/layers/")
from models.model import GNN_node

dataset = NetlistDataset(data_dir="data/example_data", load_pe = True, pl = True, processed = True)

data = dataset[0]
num_instances = data.node_congestion.shape[0]
data.num_instances = num_instances

data.edge_index_source_to_net[1] = data.edge_index_source_to_net[1] - num_instances
data.edge_index_sink_to_net[1] = data.edge_index_sink_to_net[1] - num_instances

edge_index_node_to_net = torch.cat([data.edge_index_source_to_net, data.edge_index_sink_to_net], dim=1)
edge_index_edge_weights_sink_to_netnet_to_node = torch.flip(edge_index_node_to_net, [0])

data.edge_index_source_sink = None
data.edge_index_sink_source = None

sys.path.append("models/layers/")

device = "cuda"
model = GNN_node(4, 32, 7, 1, node_dim = data.node_features.shape[1], net_dim = data.net_features.shape[1]).to(device)
data = data.to(device)

criterion_node = nn.CrossEntropyLoss()
criterion_net = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    node_representation, net_representation = model(data)
    loss_node = criterion_node(node_representation[data.train_indices], data.node_congestion[data.train_indices])
    loss_net = criterion_net(net_representation.flatten(), data.net_hpwl)
    loss = loss_node + loss_net
    loss.backward()
    optimizer.step()
    
    model.eval()
    node_representation, net_representation = model(data)
    val_loss_node = criterion_node(node_representation[data.valid_indices], data.node_congestion[data.valid_indices])
    val_loss_net = criterion_net(net_representation.flatten()[data.net_valid_indices], data.net_hpwl[data.net_valid_indices])
    print(val_loss_node.item(), val_loss_net.item())