import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch.nn import Sequential as Seq, Linear, ReLU
import torch.nn.functional as F
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

from torch_geometric.nn.conv import GATv2Conv, GCNConv, SimpleConv

class HyperConvLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, aggr='add', att=False):
        super().__init__(aggr=aggr)

        self.lin_node = Seq(Linear(in_channels, out_channels))
        self.lin_net = Seq(Linear(in_channels, out_channels))

        self.psi = Seq(Linear(out_channels * 3, out_channels),
                        ReLU(),
                        Linear(out_channels, out_channels))
        
        self.mlp = Seq(Linear(out_channels * 3, out_channels),
                        ReLU(),
                        Linear(out_channels, out_channels))

        if att:
            self.forward_conv = GATv2Conv(out_channels, out_channels, heads=2, concat=False, add_self_loops=False)
            self.back_conv = GATv2Conv(out_channels, out_channels, heads=2, concat=False, add_self_loops=False)
        else:
            self.forward_conv = SimpleConv()
            self.back_conv = SimpleConv()

        self.att = att
        
    def forward(self, x, x_net, edge_index_node_to_net, edge_weight_node_to_net, edge_type_node_to_net, edge_index_net_to_node, edge_weight_net_to_node, device):
        h_net = self.lin_net(x_net) + x_net
        h = self.lin_node(x) + x

        if self.att:
            source_mask = edge_type_node_to_net == 1
            h_net_source = self.forward_conv((h, h_net), edge_index_node_to_net[:, source_mask].to(device)) + h_net 
            sink_mask = edge_type_node_to_net == 0
            h_net_sink = self.forward_conv((h, h_net), edge_index_node_to_net[:, sink_mask].to(device)) + h_net 
            h_net = self.psi(torch.concat([h_net, h_net_sink, h_net_source], dim=1)) + x_net

            h_source = self.back_conv((h_net, h), edge_index_net_to_node[:, source_mask].to(device)) + h 
            h_sink = self.back_conv((h_net, h), edge_index_net_to_node[:, sink_mask].to(device)) + h 
            h = self.mlp(torch.concat([h, h_sink, h_source], dim=1)) + x
        
        else:
            source_mask = edge_type_node_to_net == 1
            h_net_source = self.forward_conv((h, h_net), edge_index_node_to_net[:, source_mask].to(device), edge_weight_node_to_net[source_mask].to(device)) + h_net 
            sink_mask = edge_type_node_to_net == 0
            h_net_sink = self.forward_conv((h, h_net), edge_index_node_to_net[:, sink_mask].to(device), edge_weight_node_to_net[sink_mask].to(device)) + h_net 
            h_net = self.psi(torch.concat([h_net, h_net_sink, h_net_source], dim=1)) + x_net

            h_source = self.back_conv((h_net, h), edge_index_net_to_node[:, source_mask].to(device), edge_weight_net_to_node[source_mask].to(device)) + h 
            h_sink = self.back_conv((h_net, h), edge_index_net_to_node[:, sink_mask].to(device), edge_weight_net_to_node[sink_mask].to(device)) + h 
            h = self.mlp(torch.concat([h, h_sink, h_source], dim=1)) + x
        
        return h, h_net