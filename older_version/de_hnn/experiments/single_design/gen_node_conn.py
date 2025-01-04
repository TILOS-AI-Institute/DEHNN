import torch
import torch.nn as nn


import math
import pickle
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from pyg_dataset import pyg_dataset
from tqdm import tqdm

def node_pairs_among(nodes, max_cap=5):
    us = []
    vs = []
    if max_cap == -1 or len(nodes) <= max_cap:
        for u in nodes:
            for v in nodes:
                if u == v:
                    continue
                us.append(u)
                vs.append(v)
    else:
        for u in nodes:
            vs_ = np.random.permutation(nodes)
            left = max_cap - 1
            for v_ in vs_:
                if left == 0:
                    break
                if u == v_:
                    continue
                us.append(u)
                vs.append(v_)
                left -= 1
    return us, vs


for idx in range(0, 1):
    graph_index = idx
    data_dir = "../../data/2023-03-06_data/"
    # Read node features
    file_name = data_dir + '/' + str(graph_index) + '.node_features.pkl'
    f = open(file_name, 'rb')
    dictionary = pickle.load(f)
    f.close()
    num_instances = dictionary['num_instances']
    num_nets = dictionary['num_nets']
    instance_features = torch.Tensor(dictionary['instance_features'])
    
    file_name = data_dir + '/' + str(graph_index) + '.bipartite.pkl'
    f = open(file_name, 'rb')
    dictionary = pickle.load(f)
    f.close()

    instance_idx = dictionary['instance_idx']
    net_idx = dictionary['net_idx']
    edge_dir = dictionary['edge_dir']

    net_idx = net_idx + num_instances

    net_dict = defaultdict(list)
    for i in range(len(instance_idx)):
        instance = instance_idx[i]
        net = net_idx[i]
        net_dict[net].append(instance)
        
    us4, vs4 = [], []

    for net, node_lst in tqdm(net_dict.items()):
        if len(node_lst) >= 100:
            continue

        tp = node_pairs_among(node_lst)
        us4 += tp[0]
        vs4 += tp[1]
        
    conn_dict = {"nn_edge_index": torch.tensor([us4, vs4]).long()}
    
    with open(f"../../data/2023-03-06_data/{idx}.nn_conn.pkl", "wb") as f:
        pickle.dump(conn_dict, f)
