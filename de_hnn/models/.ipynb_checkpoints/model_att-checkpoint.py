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

import sys
sys.path.append("./layers/")
from dehnn_layers import HyperConvLayer
from dgnn_layers import DiGraphConvLayer

from torch_geometric.utils.dropout import dropout_edge


class GNN_node(torch.nn.Module):
    """
    Output:
        node representations
    """
    def __init__(self, num_layer, emb_dim, out_node_dim, out_net_dim, JK = "concat", residual = True, gnn_type = 'dehnn', norm_type = "layer",
                        aggr = "add", 
                        node_dim = None, 
                        net_dim = None, 
                        num_nodes = None, # Number of nodes
                        vn = False, 
                        trans = False, 
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
        self.num_nodes = num_nodes
        self.emb_dim = emb_dim
        self.out_node_dim = out_node_dim
        self.out_net_dim = out_net_dim
        self.gnn_type = gnn_type
        self.vn = vn
        self.trans = trans
        
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

        if self.vn:
            if self.trans:
                self.transformer_virtualnode_list = torch.nn.ModuleList()
            
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
                            torch.nn.Linear(emb_dim*2, emb_dim), 
                            torch.nn.LeakyReLU(),
                            torch.nn.Linear(emb_dim, emb_dim)
                        )
                )
                
                self.back_virtualnode_list.append(
                        torch.nn.Sequential(
                            torch.nn.Linear(emb_dim*2, emb_dim), 
                            torch.nn.LeakyReLU(),
                            torch.nn.Linear(emb_dim, emb_dim)
                        )
                )

                if self.trans:
                    self.transformer_virtualnode_list.append(
                            nn.TransformerEncoderLayer(d_model=emb_dim*2, nhead=8, dim_feedforward=512)
                    )


        for layer in range(num_layer):
            if gnn_type == 'digcn':
                self.convs.append(DiGraphConvLayer(emb_dim, emb_dim, aggr=aggr))
            elif gnn_type == 'digat':
                self.convs.append(DiGraphConvLayer(emb_dim, emb_dim, aggr=aggr, att=True))
            elif gnn_type == 'dehnn':
                self.convs.append(HyperConvLayer(emb_dim, emb_dim, aggr=aggr))
            elif gnn_type == 'dehnn_att':
                self.convs.append(HyperConvLayer(emb_dim, emb_dim, aggr=aggr, att=True))
            
            if norm_type == "batch":
                self.norms.append(torch.nn.BatchNorm1d(emb_dim))
            elif norm_type == "layer":
                self.norms.append(torch.nn.LayerNorm(emb_dim))
            else:
                raise NotImplemented

        self.fc1_node = torch.nn.Linear(emb_dim, 256)
        self.fc2_node = torch.nn.Linear(256, self.out_node_dim)

        self.fc1_net = torch.nn.Linear(emb_dim, 256)
        self.fc2_net = torch.nn.Linear(256, self.out_net_dim)


    def forward(self, data, device):
        h_inst = data['node'].x.to(device)
        h_net = data['net'].x.to(device)
        
        edge_index_node_to_net, edge_weight_node_to_net = data['node', 'to', 'net'].edge_index, data['node', 'to', 'net'].edge_weight
        edge_index_net_to_node, edge_weight_net_to_node = data['net', 'to', 'node'].edge_index, data['net', 'to', 'node'].edge_weight

        edge_index_node_to_net, edge_mask = dropout_edge(edge_index_node_to_net, p = 0.2)
        edge_index_net_to_node = edge_index_net_to_node[:, edge_mask]
        
        edge_weight_node_to_net = edge_weight_node_to_net[edge_mask]
        edge_weight_net_to_node = edge_weight_net_to_node[edge_mask]
        
        edge_type_node_to_net = data['node', 'to', 'net'].edge_type[edge_mask]
        
        num_instances = data.num_instances
        
        h_inst = self.node_encoder(h_inst) 
        h_net = self.net_encoder(h_net) 
        
        if self.vn:
            batch = data.batch.to(device)
            virtualnode_embedding = self.virtualnode_encoder(data.vn.to(device))
            
        for layer in range(self.num_layer):
            if self.vn:
                h_inst = self.back_virtualnode_list[layer](torch.concat([h_inst, virtualnode_embedding[batch]], dim=1)) + h_inst
               
            h_inst, h_net = self.convs[layer](h_inst, h_net, edge_index_node_to_net, edge_weight_node_to_net, edge_type_node_to_net, edge_index_net_to_node, edge_weight_net_to_node, device)
            h_inst = self.norms[layer](h_inst)
            h_net = self.norms[layer](h_net)
            h_inst = torch.nn.functional.leaky_relu(h_inst)
            h_net = torch.nn.functional.leaky_relu(h_net)
            
            if (layer < self.num_layer - 1) and self.vn:
                virtualnode_embedding_temp = torch.concat([global_mean_pool(h_inst, batch), global_max_pool(h_inst, batch)], dim=1)
                
                if self.trans:
                    virtualnode_embedding_temp = self.transformer_virtualnode_list[layer](virtualnode_embedding_temp) 
                    virtualnode_embedding = self.mlp_virtualnode_list[layer](virtualnode_embedding_temp) + virtualnode_embedding
                else:
                    virtualnode_embedding = self.mlp_virtualnode_list[layer](virtualnode_embedding_temp) + virtualnode_embedding
            
        h_inst = self.fc2_node(torch.nn.functional.leaky_relu(self.fc1_node(h_inst)))
        h_net = self.fc2_net(torch.nn.functional.leaky_relu(self.fc1_net(h_net)))
        return h_inst, h_net
        
