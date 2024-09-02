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

from torch_geometric.utils.dropout import dropout_edge

class GNN_node(torch.nn.Module):
    """
    Output:
        node representations
    """
    def __init__(self, num_layer, emb_dim, out_node_dim, out_net_dim, JK = "concat", residual = True, gnn_type = 'dehnn', norm_type = "layer",
                        aggregators = ['mean', 'min', 'max', 'std'], # For PNA
                        scalers = ['identity', 'amplification', 'attenuation'], # For PNA
                        deg = None, # For PNA
                        edge_dim = None, # For PNA
                        use_signnet = False, # For SignNet position encoding
                        node_dim = None, 
                        net_dim = None, 
                        cfg_posenc = None, # For SignNet position encoding
                        num_nodes = None, # Number of nodes
                        vn = False, 
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
    
        self.use_signnet = use_signnet
        self.cfg_posenc = cfg_posenc

        self.vn = vn
        
        if use_signnet == False:
            self.node_encoder = nn.Sequential(
                    nn.Linear(node_dim, emb_dim),
                    nn.LeakyReLU(negative_slope = 0.1),
                    nn.Linear(emb_dim, emb_dim),
                    nn.LeakyReLU(negative_slope = 0.1)
            )

            self.net_encoder = nn.Sequential(
                    nn.Linear(net_dim, emb_dim),
                    nn.LeakyReLU(negative_slope = 0.1)
            )
            
        else:
            self.node_encoder = SignNetNodeEncoder(cfg = cfg_posenc, dim_in = node_dim, dim_emb = emb_dim, expand_x = True)

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")
                
        ###List of GNNs
        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()

        if self.vn:
            self.virtualnode_embedding = torch.nn.Embedding(1, emb_dim)   
            torch.nn.init.constant_(self.virtualnode_embedding.weight.data, 0)
            
            # self.virtualnode_embedding_top = torch.nn.Embedding(1, emb_dim)
            # torch.nn.init.constant_(self.virtualnode_embedding_top.weight.data, 0)

            self.mlp_virtualnode_list = torch.nn.ModuleList()
            #self.top_virtualnode_list = torch.nn.ModuleList()

            self.virtualnode_to_local_list = torch.nn.ModuleList()

            for layer in range(num_layer - 1):
                self.mlp_virtualnode_list.append(
                        torch.nn.Sequential(
                            torch.nn.Linear(emb_dim, emb_dim), 
                            torch.nn.LeakyReLU(negative_slope = 0.1),
                            torch.nn.Linear(emb_dim, emb_dim),
                            torch.nn.LeakyReLU(negative_slope = 0.1)
                        )
                )
                
                # self.top_virtualnode_list.append(
                #         torch.nn.Sequential(
                #             torch.nn.Linear(emb_dim, emb_dim),
                #             torch.nn.LeakyReLU(negative_slope = 0.1),
                #             torch.nn.Linear(emb_dim, emb_dim),
                #             torch.nn.LeakyReLU(negative_slope = 0.1)
                #         )
                # )

        for layer in range(num_layer):
            if gnn_type == 'gat':
                self.convs.append(GATv2Conv(in_channels = emb_dim, out_channels = emb_dim, heads = 3))
            elif gnn_type == 'gcn':
                self.convs.append(GCNConv(emb_dim, emb_dim))
            elif gnn_type == 'dehnn':
                self.convs.append(HyperConvLayer(emb_dim, emb_dim))
            if norm_type == "batch":
                self.norms.append(torch.nn.BatchNorm1d(emb_dim))
            elif norm_type == "layer":
                self.norms.append(torch.nn.LayerNorm(emb_dim))
            else:
                raise NotImplemented


        if self.JK == "concat":
            self.fc1_node = torch.nn.Linear((self.num_layer + 1) * emb_dim, 256)
            self.fc2_node = torch.nn.Linear(256, self.out_node_dim)

            self.fc1_net = torch.nn.Linear((self.num_layer + 1) * emb_dim, 64)
            self.fc2_net = torch.nn.Linear(64, self.out_net_dim)
        else:
            self.fc1_node = torch.nn.Linear(emb_dim, 256)
            self.fc2_node = torch.nn.Linear(256, self.out_node_dim)

            self.fc1_net = torch.nn.Linear(emb_dim, 64)
            self.fc2_net = torch.nn.Linear(64, self.out_net_dim)

        

    def forward(self, data, device):
        node_features, net_features, edge_index_sink_to_net, edge_weight_sink_to_net, edge_index_source_to_net, batch, num_vn = data['node'].x.to(device), data['net'].x.to(device), data['node', 'as_a_sink_of', 'net'].edge_index, data['node', 'as_a_sink_of', 'net'].edge_weight, data['node', 'as_a_source_of', 'net'].edge_index.to(device), data.batch.to(device), data.num_vn

        edge_index_sink_to_net, edge_mask = dropout_edge(edge_index_sink_to_net, p = 0.4)
        edge_index_sink_to_net = edge_index_sink_to_net.to(device)
        edge_weight_sink_to_net = edge_weight_sink_to_net[edge_mask].to(device)
        
        num_instances = data.num_instances
        
        h_list = [self.node_encoder(node_features)]
        h_net_list = [self.net_encoder(net_features)]

        if self.vn:
            virtualnode_embedding = self.virtualnode_embedding(torch.zeros(num_vn).to(batch.dtype).to(batch.device))
            #top_embedding = self.virtualnode_embedding_top(torch.zeros(num_top_vn).to(top_batch.dtype).to(top_batch.device))

        for layer in range(self.num_layer):
            if self.vn:
                h_list[layer] = h_list[layer] + virtualnode_embedding[batch]
            
            h_inst, h_net = self.convs[layer](h_list[layer], h_net_list[layer], edge_index_source_to_net, edge_index_sink_to_net, edge_weight_sink_to_net)
            h_list.append(h_inst)
            h_net_list.append(h_net)

            if (layer < self.num_layer - 1) and self.vn:
                virtualnode_embedding_temp = global_mean_pool(h_list[layer], batch) + virtualnode_embedding #global_mean_pool(h_list[layer], batch)
                virtualnode_embedding = virtualnode_embedding + self.mlp_virtualnode_list[layer](virtualnode_embedding_temp)
                #top_embedding_temp = global_mean_pool(virtualnode_embedding, top_batch) + top_embedding
        
        node_representation = torch.cat(h_list, dim = 1)
        net_representation = torch.cat(h_net_list, dim = 1)

        node_representation = torch.nn.functional.leaky_relu(self.fc2_node(torch.nn.functional.leaky_relu(self.fc1_node(node_representation), negative_slope = 0.1)), negative_slope = 0.1)
        net_representation = torch.abs(torch.nn.functional.leaky_relu(self.fc2_net(torch.nn.functional.leaky_relu(self.fc1_net(net_representation), negative_slope = 0.1)), negative_slope = 0.1))

        return node_representation, net_representation
        
