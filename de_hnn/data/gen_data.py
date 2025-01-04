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

data_dir = "cross_design_data_updated/"

to_gen = sys.argv[1:]

print("to-gen list: " + ", ".join(to_gen))

for design_fp in tqdm(os.listdir(data_dir)):
    print(f"processing {design_fp}")
    
    try: 
        if "eigen" in to_gen:
            with open(os.path.join(data_dir, design_fp, 'net2sink_nodes.pkl'), 'rb') as f:
                net2sink = pickle.load(f)
    
            with open(os.path.join(data_dir, design_fp, 'net2source_node.pkl'), 'rb') as f:
                net2source = pickle.load(f)

    except:
        print(f"read file failed for: {design_fp}")
        continue
    
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
        
        k = 5
        evals, evects = eigsh(L, k = k, which='SM')
    
        eig_fp = os.path.join(data_dir, design_fp, 'eigen.' + str(k) + '.pkl')
    
        with open(eig_fp, "wb") as f:
            dictionary = {
                'evals': evals,
                'evects': evects
            }
            pickle.dump(dictionary, f)

    if "random" in to_gen:
        print("generating random features")
        num_instances = len(node_congestion)
        num_nets = len(net_hpwl)
        
        node_ramdom_features = torch.rand(num_instances, 10)
        net_random_features = torch.rand(num_nets, 10)
        
        random_fp = os.path.join(data_dir, design_fp, 'random' + '.pkl')
        with open(random_fp, "wb") as f:
            dictionary = {
                'node_random': node_ramdom_features,
                'net_random': net_random_features
            }
            pickle.dump(dictionary, f)
