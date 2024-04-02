import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, Adagrad
from torch import optim

import numpy as np
import os
import time
import argparse
import scipy
import pickle

# Our data loader
from torch_geometric.loader import DataLoader
from pyg_dataset import pyg_dataset

# NetlistGNN data loader
from load_netlistgnn_data import load_data

# Deep Graph Library (DGL)
import dgl
import dgl.function as function
from dgl import add_self_loop, metis_partition
from typing import Tuple, List, Dict

def fo_average(g):
    degrees = g.out_degrees(g.nodes()).type(torch.float32)
    g.ndata['addnlfeat'] = (g.ndata['feat']) / degrees.view(-1, 1)
    g.ndata['inter'] = torch.zeros_like(g.ndata['feat'])
    g.ndata['wts'] = torch.ones(g.number_of_nodes()) / degrees
    g.ndata['wtmsg'] = torch.zeros_like(g.ndata['wts'])
    g.update_all(message_func=function.copy_u('addnlfeat', 'inter'),
                 reduce_func=function.sum(msg='inter', out='addnlfeat'))
    g.update_all(message_func=function.copy_u('wts', 'wtmsg'),
                 reduce_func=function.sum(msg='wtmsg', out='wts'))
    hop1 = g.ndata['addnlfeat'] / (g.ndata['wts'].view(-1, 1))
    return hop1

# Functionality to convert from our data to DGL format
def convert_to_dgl(pyg_data):
    # Number of instances
#     assert pyg_data.y.size(0) == num_nets
#     assert pyg_data.evects.size(0) == num_instances + num_nets
    
    # Compute the net's degree
    net_degree = torch.tensor(pyg_data.net_degrees)[0].unsqueeze(dim=1)

    cell_degree = torch.tensor(pyg_data.cell_degrees)[0].unsqueeze(dim=1)
    
    num_instances = pyg_data.x.size(0)
    
    # Number of nets
    num_nets = net_degree.size(0)
    
    # Instance indices
    instance_idx = pyg_data.edge_index_node_net[0, :]

    # Net indices
    net_idx = pyg_data.edge_index_node_net[1, :]

    # Make net indices to be starting from 0 again
    net_idx = net_idx - num_instances

    # Number of edges
    num_edges = net_idx.numel()

    # Checking the quality
    assert(torch.min(instance_idx).item() == 0)
    assert(torch.max(instance_idx).item() == num_instances - 1)
    assert(torch.min(net_idx).item() == 0)
    assert(torch.max(net_idx).item() == num_nets - 1)

    # Compute the near relationship
    near_idx_1 = pyg_data.edge_index_node_node[0, :]
    near_idx_2 = pyg_data.edge_index_node_node[1, :]
    near_features = torch.zeros(near_idx_1.size(0), 1)
  
    n_node = num_instances
    node_hv = torch.cat([pyg_data.x, cell_degree], dim=1)
    
    homo_graph = add_self_loop(dgl.graph((near_idx_1, near_idx_2), num_nodes=n_node))
    homo_graph.ndata['feat'] = node_hv[:n_node, :]
    extra = fo_average(homo_graph)
    
    # Net label
    net_label = torch.zeros(num_nets)
    
    # Deep Graph Library
    dgl_data = dgl.heterograph(
        {
            ('node', 'near', 'node'): (near_idx_1, near_idx_2),
            ('node', 'pins', 'net'): (instance_idx, net_idx),
            ('net', 'pinned', 'node'): (net_idx, instance_idx),
        }, 
        num_nodes_dict = {
            'node': num_instances, 
            'net': num_nets
        }
    )
    
    # Node & Edge features
    dgl_data.nodes['node'].data['hv'] = torch.cat([homo_graph.ndata['feat'], extra, pyg_data.evects[ : num_instances, :], pyg_data.evects[ : num_instances, :]], dim=1)
    #dgl_data.nodes['node'].data['pos_code'] = pyg_data.evects[ : num_instances, :]
    dgl_data.nodes['net'].data['hv'] = net_degree
    dgl_data.nodes['net'].data['degree'] = net_degree
    dgl_data.nodes['net'].data['label'] = net_label
    dgl_data.edges['pins'].data['he'] = torch.ones(net_idx.size(0), 1)
    dgl_data.edges['pinned'].data['he'] = torch.ones(net_idx.size(0), 1)
    dgl_data.edges['near'].data['he'] = near_features

    return dgl_data

