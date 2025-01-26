import sys
import torch
import math
import torch.nn as nn
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch.nn import Sequential as Seq, Linear, ReLU

from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.typing import (
    Adj,
    OptPairTensor,
    OptTensor,
    SparseTensor,
    torch_sparse,
)
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils import add_self_loops as add_self_loops_fn
from torch_geometric.utils import (
    is_torch_sparse_tensor,
    scatter,
    spmm,
    to_edge_index,
)

from utils import *

from unet import UNet

sys.path.append("./layers/")
from dehnn_layers import HyperConvLayer

from torch_geometric.utils.dropout import dropout_edge
from torch_geometric.nn.conv import GCNConv, GATv2Conv


class GNN_node(torch.nn.Module):
    """
    Output:
        node representations
    """
    def __init__(self, num_layer, emb_dim, out_node_dim, out_net_dim, JK = "concat", residual = True, gnn_type = 'dehnn', norm_type = "layer",
                        aggr = "add", 
                        scalers = ['identity', 'amplification', 'attenuation'], # For PNA
                        deg = None, # For PNA
                        edge_dim = None, # For PNA
                        node_dim = None, 
                        net_dim = None, 
                        num_nodes = None, # Number of nodes
                        vn = False, 
                        cv = False, 
                        device = 'cuda'
                    ):
        '''
            emb_dim (int): node embedding dimensionality
            num_layer (int): number of GNN message passing layers
        '''

        super(GNN_node, self).__init__()
        self.device = device

        self.num_layer = num_layer
        self.JK = JK
        self.residual = residual

        self.node_dim = node_dim
        self.net_dim = net_dim
        self.edge_dim = edge_dim
        self.num_nodes = num_nodes
        self.emb_dim = emb_dim
        self.out_node_dim = out_node_dim
        self.out_net_dim = out_net_dim
        self.gnn_type = gnn_type
        self.vn = vn
        self.cv = cv
        
        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")
                
        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        self.cv_virtualnode_list = torch.nn.ModuleList()
        self.mlp_virtualnode_list = torch.nn.ModuleList()

        for layer in range(num_layer):
            self.cv_virtualnode_list.append(
                    UNet(n_channels=emb_dim, n_classes=emb_dim, input_type=2)
            )
            
            self.mlp_virtualnode_list.append(
                    torch.nn.Sequential(
                        torch.nn.Linear(emb_dim, emb_dim), 
                        torch.nn.LeakyReLU(),
                        torch.nn.Linear(emb_dim, emb_dim)
                    )
            )
        
        self.virtualnode_encoder = nn.Sequential(
                nn.Linear(node_dim, emb_dim),
                nn.LeakyReLU(),
                nn.Linear(emb_dim, emb_dim)
        )

        self.fc1_node = torch.nn.Linear(emb_dim, 128)
        self.fc2_node = torch.nn.Linear(128, self.out_node_dim)
                                        
        self.fc1_vn = torch.nn.Linear(emb_dim, 128)
        self.fc2_vn = torch.nn.Linear(128, self.out_node_dim)


    def forward(self, data, device):
        h_inst = data['node'].x.to(device)
        h_net = data['net'].x.to(device)
        pos_lst = data.pos_lst

        num_features = h_inst.shape[1]
        num_instances = data.num_instances
        num_sites_x = data.num_sites_x
        num_sites_y = data.num_sites_y
        converter = UtilMapConverter(pos_lst, num_sites_x, num_sites_y, device)

        virtualnode_embedding_temp = converter.to_util_map(h_inst, 'amax').view(num_sites_x*num_sites_y, num_features)
        virtualnode_embedding = self.virtualnode_encoder(virtualnode_embedding_temp) 
        
        for layer in range(self.num_layer):
            virtualnode_embedding = self.mlp_virtualnode_list[layer](virtualnode_embedding) + virtualnode_embedding
            virtualnode_embedding = self.cv_virtualnode_list[layer](virtualnode_embedding.view(1, self.emb_dim, num_sites_y, num_sites_x)).view(num_sites_x*num_sites_y, self.emb_dim) + virtualnode_embedding

        h_vn = self.fc2_vn(torch.nn.functional.leaky_relu(self.fc1_vn(virtualnode_embedding)))
        h_inst = self.fc2_node(torch.nn.functional.leaky_relu(self.fc1_node(converter.to_node_features(virtualnode_embedding))))

        return h_inst, None, h_vn