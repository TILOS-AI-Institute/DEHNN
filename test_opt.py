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
from tqdm import tqdm

import matplotlib.pyplot as plt
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool

import sys
sys.path.insert(1, 'data/')
from pyg_dataset import NetlistDataset
from gen_h_dataset import process_vlsi_dataset
from data.utils import compute_degrees

sys.path.append("models/")
sys.path.append("models/layers/")
from models.model import GNN_node
from torch_geometric.utils import scatter

from typing import Dict, Tuple


def main():
    device = "cuda"
    num_sites_x, num_sites_y = 206, 300

    num_layer = 4
    num_dim = 32
    aggr = "add"
    device = "cuda"
    
    learning_rate = 0.0005
    
    configs = [
        #{'vn': False, 'cv': False, 'model_type': 'dehnn'},
        #{'vn': True, 'cv': False, 'model_type': 'dehnn'},
        {'vn': True, 'cv': True, 'model_type': 'dehnn'},
        #{'vn': True, 'cv': True, 'model_type': 'unet'}
    ]
    
    load_indices = np.array(['221'])
    test = False # if only test but not train
    restart = True # if restart training
    reload_dataset = False # if reload already processed h_dataset
    target_data_dir = "data/target_data"
    
    dataset = NetlistDataset(data_dir="data/all_designs_netlist_data", 
                            load_pe=True, pl=True, processed=True, 
                            load_indices=load_indices)
    h_dataset = process_vlsi_dataset(dataset, target_data_dir)
    # h_dataset = torch.load("h_dataset.pt")
    
    for config in configs:
        model_path = f"{config['model_type']}_{num_layer}_{num_dim}_{config['vn']}_{config['cv']}_model.pt"
        if not os.path.exists(model_path):
            continue
            
        model = torch.load(model_path).to(device)
        print(f"\nEvaluating configuration: {config}")

        h_data = h_dataset[0]
        data = dataset[0]
        model.eval()  
        for param in model.parameters():
            param.requires_grad = False

        edge_index = torch.cat([data.edge_index_source_to_net, data.edge_index_sink_to_net], dim=1)
        source_nodes = edge_index[0].to(device)
        target_nodes = edge_index[1].to(device)
        num_sites_x = 206
        num_sites_y = 300

        for epoch in range(100):
            for inner_idx in range(len(h_data.variant_data_lst)):
                pos, pos_lst_net, edge_attr, target_node, target_net = h_data.variant_data_lst[inner_idx]
                pos_lst = pos.detach().clone().requires_grad_(True).to(device)
                pos_lst.retain_grad() 
                node_pos_lst = pos_lst[data.edge_index_sink_to_net[0]]
                net_pos_lst = pos_lst[data.edge_index_source_to_net[0]][data.edge_index_sink_to_net[1]]
                edge_attr = torch.sum(torch.abs(node_pos_lst - net_pos_lst), dim=1)
                values_per_edge = pos_lst[source_nodes]
                pos_lst_net = scatter(values_per_edge, target_nodes, dim=0, reduce="mean")
                h_data['node'].x = torch.cat([h_data['node'].node_features.to(device), pos_lst], dim=1)
                h_data['net'].x = torch.cat([h_data['net'].net_features.to(device), pos_lst_net], dim=1)
                h_data['node', 'as_a_sink_of', 'net'].edge_attr = edge_attr
                h_data.pos_lst = pos_lst
                h_data.num_sites_x = num_sites_x
                h_data.num_sites_y = num_sites_y
                h_inst, h_net, virtualnode_embedding = model(h_data, device)
                h_inst.sum().backward()
                pos_lst_grad = pos_lst.grad
                
if __name__ == "__main__":
    main()
