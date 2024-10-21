import os
import shutil
import numpy as np
import pickle
import torch
import torch.nn
from torch_geometric.data import Dataset
from torch_geometric.data import Data

from sklearn.model_selection import train_test_split
from tqdm import tqdm

from collections import defaultdict
from utils import *

from pyg_dataset import NetlistDataset
from torch_geometric.data import Data, HeteroData
from torch_geometric.nn.conv.gcn_conv import gcn_norm


data_dir = "cross_design_data_updated"
load_indices = None
pl = True
load_pe = True
density = True

dataset = NetlistDataset(data_dir="cross_design_data_updated", load_pe = True, pl = True, processed = True, load_indices=None)

all_files = np.array(os.listdir(data_dir))

data_dic = {}

for data in dataset:
    data_dic[data.design_name.split("_")[1]] = data

h_dataset = []
    
for design_fp in tqdm(all_files):
    design_num = design_fp.split("_")[1]
    data = data_dic[design_num]
    
    with open(os.path.join(data_dir, design_fp, 'node_loc_x.pkl'), 'rb') as f:
        node_loc_x = pickle.load(f)
    with open(os.path.join(data_dir, design_fp, 'node_loc_y.pkl'), 'rb') as f:
        node_loc_y = pickle.load(f)
    with open(os.path.join(data_dir, design_fp, 'target_net_hpwl.pkl'), 'rb') as f:
        net_hpwl = pickle.load(f)
    with open(os.path.join(data_dir, design_fp, 'target_node_congestion_level.pkl'), 'rb') as f:
        node_congestion = pickle.load(f)

    if node_loc_x.shape[0] != 633162:
        continue

    if np.mean(node_congestion) == 0:
        continue

    file_name = os.path.join(data_dir, design_fp, 'pl_part_dict.pkl')
    f = open(file_name, 'rb')
    part_dict = pickle.load(f)
    f.close()
    batch = [part_dict[idx] for idx in range(node_congestion.shape[0])]
    num_vn = len(np.unique(batch))
    batch = torch.tensor(batch).long()
    
    node_congestion = torch.tensor(node_congestion)
    net_hpwl = torch.tensor(net_hpwl).float()
    
    node_congestion = (node_congestion >= 3).long()
    num_instances = node_congestion.shape[0]
    
    pos_lst = torch.tensor(np.vstack([node_loc_x, node_loc_y]).T)
    
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
    h_data['node'].y = node_congestion
    
    h_data['net'].x = data.net_features
    h_data['net'].y = net_hpwl
    
    h_data['node', 'as_a_sink_of', 'net'].edge_index, h_data['node', 'as_a_sink_of', 'net'].edge_weight = gcn_norm(data.edge_index_sink_to_net, add_self_loops=False)
    h_data['node', 'as_a_source_of', 'net'].edge_index = data.edge_index_source_to_net
    h_data['node', 'as_a_sink_of', 'net'].edge_attr = torch.sum(torch.abs(node_pos_lst - net_pos_lst), dim=1)[mask_edges]
    
    h_data.batch = batch
    h_data.num_vn = num_vn
    h_data.num_instances = num_instances
    
    h_data['node', 'as_a_sink_of', 'net'].edge_index, h_data['node', 'as_a_sink_of', 'net'].edge_weight = gcn_norm(data.edge_index_sink_to_net, add_self_loops=False)
    h_data['node', 'as_a_source_of', 'net'].edge_index = data.edge_index_source_to_net
    _, h_data['net', 'sink_to', 'node'].edge_weight = gcn_norm(torch.flip(h_data['node', 'as_a_sink_of', 'net'].edge_index, dims=[0]), add_self_loops=False)

    h_data.pos = pos_lst
    h_data.num_instances = num_instances
    h_dataset.append(h_data)

torch.save(h_dataset, f"../147_dataset.pt")
