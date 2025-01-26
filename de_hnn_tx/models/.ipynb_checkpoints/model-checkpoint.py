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


from unet import UNet

sys.path.append("./layers/")
from dehnn_layers import HyperConvLayer

from torch_geometric.utils.dropout import dropout_edge
from torch_geometric.nn.conv import GCNConv, GATv2Conv

from utils import *
from torch.utils.checkpoint import checkpoint

enable_checkpoint = False

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
        
        self.node_encoder = nn.Sequential(
                nn.Linear(node_dim, emb_dim),
                nn.LeakyReLU(),
                nn.Linear(emb_dim, emb_dim)
        )

        self.net_encoder = nn.Sequential(
                nn.Linear(net_dim, emb_dim),
                nn.LeakyReLU(),
                nn.Linear(emb_dim, emb_dim)
        )
        
        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")
                
        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        self.cv_virtualnode_list = torch.nn.ModuleList()

        if self.vn:            
            self.virtualnode_encoder = nn.Sequential(
                    nn.Linear(node_dim*2, emb_dim*2),
                    nn.LeakyReLU(),
                    nn.Linear(emb_dim*2, emb_dim)
            )

            self.mlp_virtualnode_list = torch.nn.ModuleList()
            self.back_virtualnode_list = torch.nn.ModuleList()

            for layer in range(num_layer):
                
                self.mlp_virtualnode_list.append(
                        torch.nn.Sequential(
                            torch.nn.Linear(emb_dim*2, emb_dim*2), 
                            torch.nn.LeakyReLU(),
                            torch.nn.Linear(emb_dim*2, emb_dim)
                        )
                )
                
                self.back_virtualnode_list.append(
                        torch.nn.Sequential(
                            torch.nn.Linear(emb_dim*2, emb_dim*2), 
                            torch.nn.LeakyReLU(),
                            torch.nn.Linear(emb_dim*2, emb_dim)
                        )
                )

                if self.cv and ((layer == 0) or (layer == num_layer-1)):                    
                    self.cv_virtualnode_list.append(
                        UNet(n_channels=emb_dim, n_classes=emb_dim, input_type=2)
                    )
                else:
                    self.cv_virtualnode_list.append(None)

        for layer in range(num_layer):
            if gnn_type == 'gat':
                self.convs.append(GATv2Conv(in_channels = emb_dim, out_channels = emb_dim, edge_dim = 1))
            elif gnn_type == 'gcn':
                self.convs.append(GCNConv(emb_dim, emb_dim))
            elif gnn_type == 'dehnn':
                self.convs.append(HyperConvLayer(emb_dim, emb_dim, aggr=aggr))
            if norm_type == "batch":
                self.norms.append(torch.nn.BatchNorm1d(emb_dim))
            elif norm_type == "layer":
                self.norms.append(torch.nn.LayerNorm(emb_dim))
            else:
                raise NotImplemented

        self.fc1_node = torch.nn.Linear(emb_dim, 128)
        self.fc2_node = torch.nn.Linear(128, self.out_node_dim)

        self.fc1_net = torch.nn.Linear(emb_dim, 64)
        self.fc2_net = torch.nn.Linear(64, self.out_net_dim)

        self.fc1_grid = torch.nn.Linear(emb_dim, 64)
        self.fc2_grid = torch.nn.Linear(64, self.out_net_dim)


    def forward(self, data, device):
        h_inst = data['node'].x.to(device)
        h_net = data['net'].x.to(device)
        edge_index_sink_to_net = data['node', 'as_a_sink_of', 'net'].edge_index.to(device)
        edge_index_source_to_net = data['node', 'as_a_source_of', 'net'].edge_index.to(device)
        edge_attr_sink_to_net = data['node', 'as_a_sink_of', 'net'].edge_attr.to(device)
        pos_lst = data.pos_lst

        num_features = h_inst.shape[1]
        num_instances = data.num_instances
        num_sites_x = data.num_sites_x
        num_sites_y = data.num_sites_y
        num_pixels = num_sites_x*num_sites_y
        converter = UtilMapConverter(pos_lst, num_sites_x, num_sites_y, device)

        if self.vn:
            virtualnode_embedding_temp = torch.concat([converter.to_util_map(h_inst, max_sum='sum').view(num_pixels, num_features), 
                                     converter.to_util_map(h_inst, max_sum='amax').view(num_pixels, num_features)], dim=1) 
            
            virtualnode_embedding = self.virtualnode_encoder(virtualnode_embedding_temp) #shape (num_pixels, num_features)
        
        h_inst = self.node_encoder(h_inst)
        h_net = self.net_encoder(h_net)
        
        for layer in range(self.num_layer):
            if self.vn:
                h_inst = self.back_virtualnode_list[layer](torch.concat([h_inst, converter.to_node_features(virtualnode_embedding)], dim=1))

            h_inst, h_net = self.convs[layer](h_inst, h_net, edge_index_source_to_net, edge_index_sink_to_net, edge_attr_sink_to_net)
                
            h_inst = self.norms[layer](h_inst)
            h_net = self.norms[layer](h_net)
            h_inst = torch.nn.functional.leaky_relu(h_inst)
            h_net = torch.nn.functional.leaky_relu(h_net)
            
            if self.vn:
                virtualnode_embedding_temp = torch.concat([converter.to_util_map(h_inst, max_sum='sum').view(num_pixels, self.emb_dim), 
                                     converter.to_util_map(h_inst, max_sum='amax').view(num_pixels, self.emb_dim)], dim=1)
                if self.cv and ((layer == 0) or (layer == self.num_layer-1)):
                    virtualnode_embedding = self.mlp_virtualnode_list[layer](virtualnode_embedding_temp) + virtualnode_embedding
                    virtualnode_embedding = self.cv_virtualnode_list[layer](virtualnode_embedding.view(1, self.emb_dim, num_sites_y, num_sites_x)).view(num_pixels, self.emb_dim) + virtualnode_embedding
                else:
                    virtualnode_embedding = self.mlp_virtualnode_list[layer](virtualnode_embedding_temp) + virtualnode_embedding

        if not self.vn:
            virtualnode_embedding = converter.to_util_map(h_inst, max_sum='amax').view(num_pixels, self.emb_dim)
            
        h_inst = self.fc2_node(torch.nn.functional.leaky_relu(self.fc1_node(h_inst)))    
        h_net = self.fc2_net(torch.nn.functional.leaky_relu(self.fc1_net(h_net)))
        virtualnode_embedding = self.fc2_grid(torch.nn.functional.leaky_relu(self.fc1_grid(virtualnode_embedding)))
        
        return h_inst, h_net, virtualnode_embedding