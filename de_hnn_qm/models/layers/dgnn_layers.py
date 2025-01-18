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


class DiGraphConvLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, aggr='add', att=False):
        super().__init__(aggr=aggr)

        self.lin_node = Seq(Linear(in_channels, out_channels))
        self.lin_net = Seq(Linear(in_channels, out_channels))
        
        self.psi = Seq(Linear(out_channels, out_channels),
                        ReLU(),
                        Linear(out_channels, out_channels))
        self.mlp = Seq(Linear(out_channels, out_channels),
                    ReLU(),
                    Linear(out_channels, out_channels))

        if att:
            self.forward_conv = GATv2Conv(out_channels, out_channels, heads=1, concat=False, add_self_loops=False)
            self.back_conv = GATv2Conv(out_channels, out_channels, heads=1, concat=False, add_self_loops=False)
        else:
            self.forward_conv = SimpleConv()
            self.back_conv = SimpleConv()

        self.att = att

    def forward(self, x, x_net, edge_index_node_to_net, edge_weight_node_to_net, edge_type_node_to_net, edge_index_net_to_node, edge_weight_net_to_node, device):
        h_net = self.lin_net(x_net) 
        h = self.lin_node(x) 

        if self.att:
            h_net_out = self.forward_conv((h, h_net), edge_index_node_to_net.to(device)) + h_net 
        else:
            h_net_out = self.forward_conv((h, h_net), edge_index_node_to_net.to(device), edge_weight_node_to_net.to(device)) + h_net 
        
        h_net_out = self.psi(h_net_out) + x_net

        if self.att:
            h_out = self.back_conv((h_net, h), edge_index_net_to_node.to(device)) + h 
        else:
            h_out = self.back_conv((h_net, h), edge_index_net_to_node.to(device), edge_weight_net_to_node.to(device)) + h 
        
        h_out = self.mlp(h_out) + x
        
        return h_out, h_net_out
        

        
