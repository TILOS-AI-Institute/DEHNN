import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_add_pool
from torch_geometric.utils import degree
from torch_geometric.nn.conv import GATv2Conv, HypergraphConv
from torch_geometric.nn.conv.pna_conv import PNAConv
from torch.nn import Sequential as Seq, Linear, ReLU
from hmpnn_layer import HMPNNLayer
from hnhn_layer import HNHNLayer
from hypersage_layer import HyperSAGELayer
from allset_layer import AllSetLayer
#from allset_transformer_layer import AllSetTransformerLayer


import math

# Precompute the statistics for position encodings
from posenc_stats import compute_posenc_stats

# Use SignNet encoder
from signnet_pos_encoder import SignNetNodeEncoder

### GCN convolution along the graph structure
class GCNConv(MessagePassing):
    def __init__(self, emb_dim, edge_dim):
        super(GCNConv, self).__init__(aggr='add')

        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.root_emb = torch.nn.Embedding(1, emb_dim)
        #self.edge_encoder = nn.Sequential(nn.Linear(edge_dim, emb_dim), nn.ReLU())

    def forward(self, x, edge_index):
        x = self.linear(x)
        #edge_embedding = self.edge_encoder(edge_attr)

        row, col = edge_index

        #edge_weight = torch.ones((edge_index.size(1), ), device=edge_index.device)
        deg = degree(row, x.size(0), dtype = x.dtype) + 1
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return self.propagate(edge_index, x=x, norm=norm) + F.relu(x + self.root_emb.weight) * 1./deg.view(-1,1)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * F.relu(x_j)

    def update(self, aggr_out):
        return aggr_out
    
    
class HyperConv(MessagePassing):
    def __init__(self, out_channels, edge_dim):
        super(HyperConv, self).__init__(aggr='add')


        self.phi = Seq(Linear(out_channels, out_channels),
                       ReLU(),
                       Linear(out_channels, out_channels))
        
        self.psi = Seq(Linear(out_channels, out_channels),
                       ReLU(),
                       Linear(out_channels, out_channels))

        self.mlp = Seq(Linear(out_channels * 3, out_channels * 3),
                       ReLU(),
                       Linear(out_channels * 3, out_channels))

    def forward(self, x, x_net, net_inst_adj, inst_net_adj_v_drive, inst_net_adj_v_sink):
        
        h = self.phi(F.dropout(x))
        
        net_agg = torch.mm(net_inst_adj, h) + x_net
        
        h_drive = F.dropout(torch.mm(inst_net_adj_v_drive, net_agg))
        
        h_sink = self.psi(F.dropout(torch.mm(inst_net_adj_v_sink, net_agg)))

        h = self.mlp(torch.concat([x, h_drive, h_sink], dim=1)) + x
        
        return h, x_net 

class HyperConvNoDir(MessagePassing):
    def __init__(self, out_channels, edge_dim):
        super(HyperConvNoDir, self).__init__(aggr='add')


        self.phi = Seq(Linear(out_channels, out_channels),
                       ReLU(),
                       Linear(out_channels, out_channels))

        self.mlp = Seq(Linear(out_channels * 2, out_channels * 2),
                       ReLU(),
                       Linear(out_channels * 2, out_channels))

    def forward(self, x, x_net, net_inst_adj, inst_net_adj_v_drive, inst_net_adj_v_sink):
        
        h = self.phi(x)
        
        net_agg = torch.mm(net_inst_adj, h) + x_net
        
        h_drive = torch.mm(inst_net_adj_v_drive, net_agg)
        
        h_sink = torch.mm(inst_net_adj_v_sink, net_agg)
        
        h_update = h_drive + h_sink
        
        h = self.mlp(torch.concat([x, h_update], dim=1)) + x
        
        return h, net_agg   

### GNN to generate node embedding
class GNN_node(torch.nn.Module):
    """
    Output:
        node representations
    """
    def __init__(self, num_layer, emb_dim, JK = "concat", residual = True, gnn_type = 'gin', norm_type = "layer",
                        aggregators = ['mean', 'min', 'max', 'std'], # For PNA
                        scalers = ['identity', 'amplification', 'attenuation'], # For PNA
                        deg = None, # For PNA
                        edge_dim = None, # For PNA

                        use_signnet = False, # For SignNet position encoding
                        node_dim = None, # For SignNet position encoding
                        cfg_posenc = None, # For SignNet position encoding

                        num_nodes = None, # Number of nodes
                        incidence = None,
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
        ### add residual connection or not
        self.residual = residual

        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.num_nodes = num_nodes
        self.emb_dim = emb_dim

        self.use_signnet = use_signnet
        self.cfg_posenc = cfg_posenc
        if use_signnet == False:
            self.node_encoder = nn.Sequential(
                    nn.Linear(node_dim, 2 * emb_dim),
                    nn.LeakyReLU(negative_slope = 0.1),
                    nn.Linear(2 * emb_dim, emb_dim),
                    nn.LeakyReLU(negative_slope = 0.1)
            )
        else:
            self.node_encoder = SignNetNodeEncoder(cfg = cfg_posenc, dim_in = node_dim, dim_emb = emb_dim, expand_x = True).to(device = device)

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")
                
        ###List of GNNs
        self.convs = torch.nn.ModuleList()
        self.re_convs = torch.nn.ModuleList()
        #self.batch_norms = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        #self.mlps = torch.nn.ModuleList()
        self.gnn_type = gnn_type

        for layer in range(num_layer):
            if gnn_type == 'gat':
                self.convs.append(GATv2Conv(in_channels = emb_dim, out_channels = emb_dim, heads = 1))
                self.re_convs.append(GATv2Conv(in_channels = emb_dim, out_channels = emb_dim, heads = 1))
            elif gnn_type == 'gcn':
                self.convs.append(GCNConv(emb_dim, edge_dim))
                self.re_convs.append(GCNConv(emb_dim, edge_dim))
            elif gnn_type == 'pna':
                self.convs.append(PNAConv(in_channels = emb_dim, out_channels = emb_dim, aggregators = aggregators, scalers = scalers, deg = deg, edge_dim = edge_dim))
            elif gnn_type == 'hyper':
                self.convs.append(HyperConv(emb_dim, edge_dim))
            elif gnn_type == 'hypernodir':
                self.convs.append(HyperConvNoDir(emb_dim, edge_dim))
            elif gnn_type == 'hmpnn':
                self.convs.append(HMPNNLayer(
                    emb_dim,
                    adjacency_dropout=0.7,
                    updating_dropout=0.5,
                ))
            elif gnn_type == "hnhn":
                self.convs.append(HNHNLayer(
                    in_channels=emb_dim,
                    hidden_channels=emb_dim
                ))
            elif gnn_type == "allset":
                self.convs.append(AllSetLayer(in_channels=emb_dim, hidden_channels=emb_dim))
            
            elif gnn_type == 'sota':
                self.convs.append(HypergraphConv(in_channels = emb_dim, out_channels = emb_dim, use_attention=True, concat=False))
            
            else:
                raise ValueError('Undefined GNN type called {}'.format(gnn_type))
                    
            #self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))
            if norm_type == "batch":
                self.norms.append(torch.nn.BatchNorm1d(emb_dim))
            elif norm_type == "layer":
                self.norms.append(torch.nn.LayerNorm(emb_dim))
            else:
                raise NotImplemented
                
    def forward(self, batched_data):

        if self.gnn_type not in ["gcn", "gat"]: 
            x, net_inst_adj, inst_net_adj_v_drive, inst_net_adj_v_sink, batch = batched_data.x, batched_data.net_inst_adj, batched_data.inst_net_adj_v_drive, batched_data.inst_net_adj_v_sink, batched_data.batch
        
        else:
            x, edge_index_node_net, edge_index_net_node, edge_attr, batch = batched_data.x, batched_data.edge_index_node_net, batched_data.edge_index_net_node, batched_data.edge_attr, batched_data.batch

        num_instances = batched_data.num_instances

        ### computing input node embedding
        if self.use_signnet == False:
            x = self.node_encoder(x)
            h_list = [x]
        else:
            batched_data = compute_posenc_stats(batched_data, ['SignNet'], True, self.cfg_posenc)
            batched_data.x = batched_data.x.to(device = self.device)
            batched_data.eigvecs_sn = batched_data.eigvecs_sn.to(device = self.device)

            h_list = [self.node_encoder(batched_data).x]
        

        for layer in range(self.num_layer):
            if self.gnn_type not in ['gcn', 'gat']:
                h_inst, h_net = h_list[layer][:num_instances], h_list[layer][num_instances:]
                
                if self.gnn_type == 'hmpnn':
                    h_inst, h_net = self.convs[layer](h_inst, h_net, net_inst_adj.T)
                elif self.gnn_type == 'hnhn':
                    h_inst, h_net = self.convs[layer](h_inst, incidence_1=net_inst_adj)
                elif self.gnn_type == 'allset':
                    h_inst, h_net = self.convs[layer](h_inst, net_inst_adj.T)
                elif self.gnn_type == 'sota':
                  h_inst = self.convs[layer](x=h_list[layer][:num_instances], hyperedge_index=net_inst_adj.T._indices(), hyperedge_attr=h_list[layer][num_instances:], num_edges=h_net.size(0)) 
                  h_net = torch.mm(net_inst_adj, h_inst) 
                else:
                    h_inst, h_net = self.convs[layer](h_inst, h_net, net_inst_adj, inst_net_adj_v_drive, inst_net_adj_v_sink)
                
                h = torch.cat([h_inst, h_net], dim=0)

            else:
                h = self.convs[layer](h_list[layer], edge_index_node_net)
                h_re = self.re_convs[layer](h, edge_index_net_node)
            
                h = h_re

            # Layer norm
            h = self.norms[layer](h)

            h = F.leaky_relu(h, negative_slope = 0.1)

            if self.residual:
                h += h_list[layer]

            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(self.num_layer + 1):
                node_representation += h_list[layer]
        elif self.JK == "concat":
            node_representation = torch.cat(h_list, dim = 1)

        return node_representation[:num_instances]


### Virtual GNN to generate node embedding
class GNN_node_Virtualnode(torch.nn.Module):
    """
    Output:
        node representations
    """
    def __init__(self, num_layer, emb_dim, JK = "concat", residual = True, gnn_type = 'gin', norm_type = "layer",
                        aggregators = ['mean', 'min', 'max', 'std'], # For PNA
                        scalers = ['identity', 'amplification', 'attenuation'], # For PNA
                        deg = None, # For PNA
                        edge_dim = None, # For PNA

                        use_signnet = False, # For SignNet position encoding
                        node_dim = None, # For SignNet position encoding
                        cfg_posenc = None, # For SignNet position encoding
                 
                        single = False, # Single OR Hier VN

                        num_nodes = None, # Number of nodes
               
                        device = 'cuda'
                    ):
        '''
            emb_dim (int): node embedding dimensionality
        '''

        super(GNN_node_Virtualnode, self).__init__()
        self.device = device
        
        self.single = single

        self.gnn_type = gnn_type
        self.num_layer = num_layer
        self.JK = JK
        ### add residual connection or not
        self.residual = residual

        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.num_nodes = num_nodes
        self.emb_dim = emb_dim

        self.use_signnet = use_signnet
        self.cfg_posenc = cfg_posenc
        
        if use_signnet == False:
            self.node_encoder = nn.Sequential(
                    nn.Linear(node_dim, 2 * emb_dim),
                    nn.LeakyReLU(negative_slope = 0.1),
                    nn.Linear(2 * emb_dim, emb_dim),
                    nn.LeakyReLU(negative_slope = 0.1)
            )

        else:
            self.node_encoder = SignNetNodeEncoder(cfg = cfg_posenc, dim_in = node_dim, dim_emb = emb_dim, expand_x = True).to(device = device)

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        ### set the initial virtual node embedding to 0.
        self.virtualnode_embedding = torch.nn.Embedding(1, emb_dim)   
        torch.nn.init.constant_(self.virtualnode_embedding.weight.data, 0)
        
        if not single:
            self.virtualnode_embedding_top = torch.nn.Embedding(1, emb_dim)
            torch.nn.init.constant_(self.virtualnode_embedding_top.weight.data, 0)

        ### List of GNNs
        self.convs = torch.nn.ModuleList()
        #self.batch_norms = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        #self.mlps = torch.nn.ModuleList()
        
        ### List of MLPs to transform virtual node at every layer
        self.mlp_virtualnode_list = torch.nn.ModuleList()
        if not single:
            self.top_virtualnode_list = torch.nn.ModuleList()

        for layer in range(num_layer):
            if gnn_type == 'gin':
                self.convs.append(GINConv(emb_dim, edge_dim))
            elif gnn_type == 'gcn':
                self.convs.append(GCNConv(emb_dim, edge_dim))
                self.mlps.append(torch.nn.Linear(emb_dim * 2, emb_dim))
            elif gnn_type == 'pna':
                self.convs.append(PNAConv(in_channels = emb_dim, out_channels = emb_dim, aggregators = aggregators, scalers = scalers, deg = deg, edge_dim = edge_dim))
            elif gnn_type == 'hyper':
                self.convs.append(HyperConv(emb_dim, edge_dim))
            else:
                raise ValueError('Undefined GNN type called {}'.format(gnn_type))

            #self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))
            if norm_type == "batch":
                self.norms.append(torch.nn.BatchNorm1d(emb_dim))
            elif norm_type == "layer":
                self.norms.append(torch.nn.LayerNorm(emb_dim))
            else:
                raise NotImplemented


            if gnn_type != 'hyper' and gnn_type != 'hypernodir':
                # Reverse, from target to source
                self.re_convs = torch.nn.ModuleList()

                for layer in range(num_layer):
                    if gnn_type == 'gin':
                        self.re_convs.append(GINConv(emb_dim, edge_dim))
                    elif gnn_type == 'gcn':
                        self.re_convs.append(GCNConv(emb_dim, edge_dim))
                    elif gnn_type == 'pna':
                        self.re_convs.append(PNAConv(in_channels = emb_dim, out_channels = emb_dim, aggregators = aggregators, scalers = scalers, deg = deg, edge_dim = edge_dim))
                    else:
                        raise ValueError('Undefined GNN type called {}'.format(gnn_type))
    
        for layer in range(num_layer - 1):
            self.mlp_virtualnode_list.append(
                    torch.nn.Sequential(
                        torch.nn.Linear(emb_dim, emb_dim), 
                        torch.nn.LeakyReLU(negative_slope = 0.1),
                        torch.nn.Linear(emb_dim, emb_dim), 
                        torch.nn.LeakyReLU(negative_slope = 0.1)
                    )
            )
            
            if not single:
            
                self.top_virtualnode_list.append(
                        torch.nn.Sequential(
                            torch.nn.Linear(emb_dim, emb_dim),
                            torch.nn.LeakyReLU(negative_slope = 0.1),
                            torch.nn.Linear(emb_dim, emb_dim),
                            torch.nn.LeakyReLU(negative_slope = 0.1)
                        )
                )


    def forward(self, batched_data):

        x,  net_inst_adj, inst_net_adj_v_drive, inst_net_adj_v_sink, batch, top_batch, num_vn, num_top_vn = batched_data.x, batched_data.net_inst_adj, batched_data.inst_net_adj_v_drive, batched_data.inst_net_adj_v_sink, batched_data.part_id, batched_data.top_part_id, batched_data.num_vn, batched_data.num_top_vn
        
        ### virtual node embeddings
        single = self.single
        num_instances = batched_data.num_instances
        
        if not single:
            virtualnode_embedding = self.virtualnode_embedding(torch.zeros(num_vn).to(batch.dtype).to(batch.device))
            top_embedding = self.virtualnode_embedding_top(torch.zeros(num_top_vn).to(top_batch.dtype).to(top_batch.device)) 
        else:
            batch = torch.zeros_like(batch)
            virtualnode_embedding = self.virtualnode_embedding(torch.zeros(num_vn).to(batch.dtype).to(batch.device))
        
        ### computing input node embedding
        if self.use_signnet == False:
            x = self.node_encoder(x)
            h_list = [x]

        else:
            batched_data = compute_posenc_stats(batched_data, ['SignNet'], True, self.cfg_posenc)
            batched_data.x = batched_data.x.to(device = self.device)
            batched_data.eigvecs_sn = batched_data.eigvecs_sn.to(device = self.device)

            h_list = [self.node_encoder(batched_data).x]

        for layer in range(self.num_layer):
            h_inst, h_net = h_list[layer][:num_instances], h_list[layer][num_instances:]
            ###
            if not single:
                h_inst = h_inst + (virtualnode_embedding + top_embedding[top_batch])[batch]
            else:
                h_inst = h_inst + virtualnode_embedding[batch]
            ###
            if self.gnn_type == 'hyper':
                h_inst, h_net = self.convs[layer](h_inst, h_net, net_inst_adj, inst_net_adj_v_drive, inst_net_adj_v_sink)
                h = torch.cat([h_inst, h_net], dim=0)
                            
            else:
                h = self.convs[layer](h_list[layer], edge_index_node_net, edge_attr)
                h_re = self.re_convs[layer](h_list[layer], edge_index_net_node.flip([0]), edge_attr)

                h = self.mlps[layer](torch.concat([h, h_re], dim = 1))
                
            # Layer norm
            h = self.norms[layer](h)
            h = F.leaky_relu(h, negative_slope = 0.1)

            if self.residual:
                h = h + h_list[layer]

            h_list.append(h)

            ### update the virtual nodes
            if layer < self.num_layer - 1:
                ### add message from graph nodes to virtual nodes
                virtualnode_embedding_temp = global_mean_pool(F.dropout(h_list[layer][:num_instances]), batch) + virtualnode_embedding
                ### transform virtual nodes using MLP
                if self.residual:
                    virtualnode_embedding = virtualnode_embedding + self.mlp_virtualnode_list[layer](virtualnode_embedding_temp)
                else:
                    virtualnode_embedding = self.mlp_virtualnode_list[layer](virtualnode_embedding_temp)
                
                if not single:
                    top_embedding_temp = global_mean_pool(F.dropout(virtualnode_embedding), top_batch) + top_embedding
                    if self.residual:
                        top_embedding = top_embedding + self.top_virtualnode_list[layer](top_embedding_temp)
                    else:
                        top_embedding = self.top_virtualnode_list[layer](top_embedding_temp)



        ### Different implementations of Jk-concat
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(self.num_layer + 1):
                node_representation += h_list[layer]
        elif self.JK == "concat":
            node_representation = torch.cat(h_list, dim = 1)

        return node_representation[:num_instances]


if __name__ == "__main__":
    pass

