##
# @file   congestion_prediction.py
# @author Zhili Xiong (DREAMPlaceFPGA)
# @date   Jan 2025
# @brief  The congestion prediction function
#

import os
import time
import torch
from torch import nn
from torch.autograd import Function
import pickle
from torch_geometric.data import Dataset
from torch_geometric.data import Data, HeteroData
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.utils import scatter
import random
import logging
import numpy as np
import matplotlib.pyplot as plt
import pdb
import sys

sys.path.append("/home/local/eda02/zhilix/projects/MP-GNN_bak/de_hnn_tx/")
sys.path.append("/home/local/eda02/zhilix/projects/MP-GNN_bak/de_hnn_tx/models")
sys.path.append("/home/local/eda02/zhilix/projects/MP-GNN_bak/de_hnn_tx/models/layers")
sys.path.append("/home/local/eda02/zhilix/projects/MP-GNN_bak/unet")
sys.path.append("/home/local/eda02/zhilix/projects/MP-GNN_bak/unet/models")

# Set seed for reproducibility
np.random.seed(0)
random.seed(0)
torch.manual_seed(0)
# If using CUDA, set seed for GPU as well
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)  # if you are using multi-GPU
# Optional for consistent behavior across CPU and GPU operations
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)


class CongestionPrediction(nn.Module):
    def __init__(self, params, placedb, device):
        super(CongestionPrediction, self).__init__()
        self.params = params
        self.placedb = placedb
        self.device = device
        self.num_sites_x = placedb.num_sites_x
        self.num_sites_y = placedb.num_sites_y
        self.num_nodes = placedb.num_nodes
        self.num_instances = 0
        self.num_nets = 0CongestionPrediction
        self.edge_index = None
        self.source_nodes = None
        self.target_nodes = None

        self.data = None
        self.h_data = HeteroData()

        #### initialize the netlist data ####
        self.init_netlist_data()
        #### load the DEHNN model ####
        self.dehnn_model = torch.load(params.dehnn_model).to(device)
        self.dehnn_model.train()

        self.congestion = None
        self.pos_grad = None
        self.congestion_iter = 0
    
    def init_netlist_data(self):
        """
        @brief Initialize the netlist data for DEHNN model
        
        """
        ####### load netlist information from pyg_data.pkl ######
        design_fp = self.params.design_idx + '_netlist_data'
        pyg_data_fp = os.path.join(self.params.gnn_data_dir, design_fp, 'pyg_data.pkl')
        self.data = torch.load(pyg_data_fp)

        ### ignore nets with large fanout (>=3000) ###
        out_degrees = self.data.net_features[:, 1]
        mask = (out_degrees < 3000)
        mask_edges = mask[self.data.edge_index_source_to_net[1]] 
        filtered_edge_index = self.data.edge_index_source_to_net[:, mask_edges]
        self.data.edge_index_source_to_net = filtered_edge_index
    
        mask_edges = mask[self.data.edge_index_sink_to_net[1]] 
        filtered_edge_index = self.data.edge_index_sink_to_net[:, mask_edges]
        self.data.edge_index_sink_to_net = filtered_edge_index
        
        ####### initialize the hetero data ######
        self.h_data['node'].node_features = self.data.node_features
        self.h_data['net'].net_features = self.data.net_features
        self.h_data.num_instances = self.data.node_features.shape[0]
        self.h_data['node', 'as_a_sink_of', 'net'].edge_index = self.data.edge_index_sink_to_net 
        self.h_data['node', 'as_a_source_of', 'net'].edge_index = self.data.edge_index_source_to_net
        self.h_data = self.h_data.to(self.device)

        self.num_instances = self.h_data.num_instances
        self.num_nets = self.h_data['net'].net_features.shape[0]
        self.edge_index = torch.cat([self.data.edge_index_source_to_net, self.data.edge_index_sink_to_net], dim=1).to(self.device)
        self.source_nodes = self.edge_index[0]
        self.target_nodes = self.edge_index[1] 

        pos_lst = torch.zeros(self.h_data.num_instances, 2).to(self.device)
        pos_lst_net = torch.zeros(self.num_nets, 2).to(self.device)
        self.h_data['node'].x = torch.cat([self.h_data['node'].node_features, pos_lst], dim=1)
        self.h_data['net'].x = torch.cat([self.h_data['net'].net_features, pos_lst_net], dim=1)
        self.h_data['node', 'as_a_sink_of', 'net'].edge_attr = None


    def update_netlist_data(self, pos):
        """
        @brief update node locations and the variant data list
        """
        ######## update pos_lst and edge_attr ########
        node_loc_x = pos[:self.num_instances].detach()
        node_loc_y = pos[self.num_nodes:self.num_nodes+self.num_instances].detach()

        # In fact, the require grad should be added right here, instead of x
        pos_lst = torch.cat([node_loc_x.view(-1, 1), node_loc_y.view(-1, 1)], dim=1).detach().clone().requires_grad_(True).to(device)
        pos_lst.retain_grad() 
        node_pos_lst = pos_lst[self.data.edge_index_sink_to_net[0]]
        net_pos_lst = pos_lst[self.data.edge_index_source_to_net[0]][self.data.edge_index_sink_to_net[1]]
        edge_attr = torch.sum(torch.abs(node_pos_lst - net_pos_lst), dim=1)

        values_per_edge = pos_lst[self.source_nodes]
        pos_lst_net = scatter(values_per_edge, self.target_nodes, dim=0, reduce="mean")

        # From Zhishang: No longer needed
        ######## do partition for virtual nodes based on placement ########
        # unit_width = (self.num_sites_x-1)/50
        # unit_height = (self.num_sites_y-1)/50
        
        # node2part =  (node_loc_x/unit_width).int()*50 + (node_loc_y/unit_height).int()
        # node2part = node2part.int().cpu().numpy() # convert tensor to numpy 

        # part_to_idx = {val:idx for idx, val in enumerate(np.unique(node2part))}
        # part_dict = {idx:part_to_idx[part_id] for idx, part_id in enumerate(node2part)}
        # batch = [part_dict[idx] for idx in range(self.num_instances)]
        # num_vn = len(np.unique(batch))
        # batch = torch.tensor(batch).long().to(self.device)

        # From Zhishang: No longer needed
        # vn_node = torch.concat([global_mean_pool(self.h_data['node'].node_features, batch), 
        #         global_mean_pool(pos_lst, batch), 
        #         global_max_pool(self.h_data['node'].node_features, batch), 
        #         global_max_pool(pos_lst, batch)], dim=1)

        ################# update h_data ##################
        ## avoid in-place operation ##
        
        self.h_data['node'].x = torch.cat([self.h_data['node'].node_features, pos_lst], dim=1)
        self.h_data['net'].x = torch.cat([self.h_data['net'].net_features, pos_lst_net], dim=1)

        #x_detached = self.h_data['node'].x.detach()
        #x_detached[:, -2:] = pos_lst
        #self.h_data['node'].x = x_detached
        # self.h_data['node'].x[:, -2:] = pos_lst
        #self.h_data['net'].x[:, -2:] = pos_lst_net

        self.h_data['node', 'as_a_sink_of', 'net'].edge_attr = edge_attr
        #self.h_data.batch = batch
        #self.h_data.num_vn = num_vn
        #self.h_data.vn = vn_node
        
        return pos_lst

    def dehnn_forward(self, pos):
        """
        @brief Forward function of the DEHNN model
        """
        pos_lst = self.update_netlist_data(pos)
        #self.h_data['node'].x.requires_grad = True
        features, pred = self.dehnn_model(self.h_data, pos.device)
        pred.sum().backward()

        # pos_lst_grad = self.h_data['node'].x.grad[:, -2:].clone().detach()
        # pos_lst_grad = self.h_data['node'].x.grad[:, -2:]
        pos_lst_grad = pos_lst.grad
        pos_grad = torch.zeros_like(pos)
        pos_grad[: self.num_instances] = pos_lst_grad[:, 0].clone()
        pos_grad[self.num_nodes: self.num_nodes+self.num_instances] = pos_lst_grad[:, 1].clone()

        return pred, pos_grad


    def forward(self, pos):
        """
        @brief Forward function of the congestion prediction model
        """
        self.congestion, self.pos_grad = self.dehnn_forward(pos)
        self.congestion_iter += 1

        return self.congestion.sum(), self.pos_grad


    def convert_to_util_map(self, placedb, pos_lst, node_features, device):
        """
        Convert the output from dehnn model to utilization map
        """
        min_value = torch.min(node_features).item()
        indices = (pos_lst[:, 0].long()*placedb.num_sites_y + pos_lst[:, 1].long()).to(device)
        # breakpoint()
        util_map = torch.ones(placedb.num_sites_x*placedb.num_sites_y).to(device)*min_value
        util_map.scatter_reduce_(0, indices, node_features, 'amax', include_self=True)
        util_map = util_map.view(placedb.num_sites_x, placedb.num_sites_y)

        return util_map



        