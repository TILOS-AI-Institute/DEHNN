import os
import numpy as np
import pickle
import torch
import torch.nn
import sys

from scipy.sparse.linalg import eigsh
from numpy.linalg import eigvals
from torch_geometric.utils import (get_laplacian, to_scipy_sparse_matrix, to_undirected, to_dense_adj)
from collections import defaultdict

from sklearn.model_selection import train_test_split

from tqdm import tqdm

data_dir = "cross_design_data"

to_gen = sys.argv[1:]

print("to-gen list: " + ", ".join(to_gen))

for design_fp in tqdm(os.listdir(data_dir)):
    print(f"processing {design_fp}")
    
    with open(os.path.join(data_dir, design_fp, 'net2sink_nodes.pkl'), 'rb') as f:
        net2sink = pickle.load(f)

    with open(os.path.join(data_dir, design_fp, 'net2source_node.pkl'), 'rb') as f:
        net2source = pickle.load(f)

    with open(os.path.join(data_dir, design_fp, 'target_net_hpwl.pkl'), 'rb') as f:
        net_hpwl = pickle.load(f)
    
    with open(os.path.join(data_dir, design_fp, 'target_node_congestion_level.pkl'), 'rb') as f:
        node_congestion = pickle.load(f)

    with open(os.path.join(data_dir, design_fp, 'node_loc_x.pkl'), 'rb') as f:
        node_loc_x = pickle.load(f)
    
    with open(os.path.join(data_dir, design_fp, 'node_loc_y.pkl'), 'rb') as f:
        node_loc_y = pickle.load(f)

    if "eigen" in to_gen:
        print("generating lap-eigenvectors")
        edge_index = []
    
        for net_idx in range(len(net2sink)):
            sink_idx_lst = net2sink[net_idx]
            source_idx = net2source[net_idx]
        
            for sink_idx in sink_idx_lst:
                edge_index.append([source_idx, sink_idx])
                edge_index.append([sink_idx, source_idx])
    
        edge_index = torch.tensor(edge_index).T.long()
    
        num_instances = len(node_loc_x)
    
        L = to_scipy_sparse_matrix(
            *get_laplacian(edge_index, normalization="sym", num_nodes = num_instances)
        )
        
        k = 10
        evals, evects = eigsh(L, k = k, which='SM')
    
        eig_fp = os.path.join(data_dir, design_fp, 'eigen.' + str(k) + '.pkl')
    
        with open(eig_fp, "wb") as f:
            dictionary = {
                'evals': evals,
                'evects': evects
            }
            pickle.dump(dictionary, f)

    if "split" in to_gen:
        print("generating split index")
        node_index = [idx for idx in range(len(node_congestion))]
        net_index = [idx for idx in range(len(net_hpwl))]
    
        train_val_indices, test_indices = train_test_split(node_index, test_size=0.1)
        train_num = int(len(train_val_indices)*0.8)
        train_indices, valid_indices = train_val_indices[:train_num], train_val_indices[train_num:]
    
        net_train_val_indices, net_test_indices = train_test_split(net_index, test_size=0.1)
        net_train_num = int(len(net_train_val_indices)*0.8)
        net_train_indices, net_valid_indices = net_train_val_indices[:net_train_num], net_train_val_indices[net_train_num:]
        
        dictionary = {
            'train_indices': train_indices,
            'valid_indices': valid_indices,
            'test_indices': test_indices,
            'net_train_indices': net_train_indices,
            'net_valid_indices': net_valid_indices,
            'net_test_indices': net_test_indices
        }
        split_fp = os.path.join(data_dir, design_fp, 'split.pkl')
        f = open(split_fp, "wb")
        pickle.dump(dictionary, f)
        f.close()

    if "part" in to_gen:
        print("generating placement-based partitions")
        x_lst = node_loc_x - min(node_loc_x)
        y_lst = node_loc_y - min(node_loc_y)
        unit_width = abs(max(x_lst) - min(x_lst))/100
        unit_height = abs(max(y_lst) - min(y_lst))/100
        part_dict = {}
        phy_id_set = set()
        
        for idx in range(len(x_lst)):
            x = x_lst[idx]
            y = y_lst[idx]
            x = int(x//unit_width)
            y = int(y//unit_height)
            part_id = x * 100 + y
            part_dict[idx] = part_id
            phy_id_set.add(part_id)
        
        part_to_idx = {val:idx for idx, val in enumerate(phy_id_set)}
        part_dict = {idx:part_to_idx[part_id] for idx, part_id in part_dict.items()}
        file_name = os.path.join(data_dir, design_fp, 'pl_part_dict.pkl')
        f = open(file_name, 'wb')
        pickle.dump(part_dict, f)
        f.close()

