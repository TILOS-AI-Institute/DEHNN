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
    def __init__(self, net_out_channels, out_channels, aggr='add'):
        super().__init__(aggr=aggr)

        self.phi = Seq(Linear(out_channels * 3, out_channels * 3),
                        ReLU(),
                        Linear(out_channels * 3, out_channels))

        self.psi = Seq(Linear(net_out_channels, net_out_channels),
                        ReLU(),
                        Linear(net_out_channels, net_out_channels))
        
        self.mlp = Seq(Linear(net_out_channels * 2, net_out_channels * 2),
                        ReLU(),
                        Linear(net_out_channels * 2, net_out_channels))

        self.conv = SimpleConv()
        self.forward_conv = GATv2Conv(out_channels, out_channels, edge_dim=1, heads=2, concat=False, add_self_loops=False)
        self.back_conv = GATv2Conv(out_channels, out_channels, edge_dim=1, heads=2, concat=False, add_self_loops=False)
        
    def forward(self, x, x_net, edge_index_source_to_net, edge_index_sink_to_net, edge_weight_sink_to_net, edge_attr_sink_to_net):
        h_net_source = self.conv((x, x_net), edge_index_source_to_net) + x_net
        h_net_sink = self.psi(self.forward_conv((x, x_net), edge_index_sink_to_net, edge_attr = edge_attr_sink_to_net)) + x_net
        h_net = self.mlp(torch.concat([h_net_source, h_net_sink], dim=1)) + x_net
        h = self.phi(torch.concat([x, self.conv((h_net, x), torch.flip(edge_index_source_to_net, dims=[0])), 
                                   self.back_conv((h_net, x), torch.flip(edge_index_sink_to_net, dims=[0]), edge_attr = edge_attr_sink_to_net)], dim=1)) + x
        return h, h_net

# class HyperConvLayer(MessagePassing):
#     def __init__(self, net_out_channels, out_channels, aggr='add'):
#         super().__init__(aggr=aggr)

#         self.phi = Seq(Linear(out_channels, out_channels),
#                         ReLU(),
#                         Linear(out_channels, out_channels))

#         self.psi = Seq(Linear(net_out_channels, net_out_channels),
#                         ReLU(),
#                         Linear(net_out_channels, net_out_channels))
        
#         self.mlp = Seq(Linear(net_out_channels * 3, net_out_channels * 3),
#                         ReLU(),
#                         Linear(net_out_channels * 3, net_out_channels))

#         self.conv = SimpleConv()
#         self.back_conv = GATv2Conv(out_channels, out_channels, edge_dim=1)
        
#     def forward(self, x, x_net, edge_index_source_to_net, edge_index_sink_to_net, edge_weight_sink_to_net, edge_attr_sink_to_net): 
#         h = self.phi(x) + x
#         h_net_source = self.conv((h, x_net), edge_index_source_to_net) + x_net
#         h_net_sink = self.psi(self.propagate(edge_index_sink_to_net, x=(h, x_net), edge_weight=edge_weight_sink_to_net)) + x_net
#         h_net = self.mlp(torch.concat([x_net, h_net_source, h_net_sink], dim=1)) + x_net
#         h = self.conv((h_net, h), torch.flip(edge_index_source_to_net, dims=[0])) + self.back_conv((h_net, h), torch.flip(edge_index_source_to_net, dims=[0])) + self.back_conv((h_net, h), torch.flip(edge_index_sink_to_net, dims=[0]), edge_attr = edge_attr_sink_to_net) + x
        
#         return h, h_net
