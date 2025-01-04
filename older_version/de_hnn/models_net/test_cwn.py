import torch
from cwn import CWN

in_channels_0 = 10
in_channels_1 = 20
in_channels_2 = 30
hid_channels = 40
num_classes = 100
n_layers = 3

model = CWN(
    in_channels_0 = in_channels_0,
    in_channels_1 = in_channels_1,
    in_channels_2 = in_channels_2,
    hid_channels = hid_channels,
    num_classes = num_classes,
    n_layers = n_layers,
    instance_predict = False,
    net_predict = True
)

n_nodes = 4
n_edges = 5
n_faces = 2

"""
x_0 : torch.Tensor, shape = (n_nodes, in_channels_0)
    Input features on the nodes (0-cells).
x_1 : torch.Tensor, shape = (n_edges, in_channels_1)
    Input features on the edges (1-cells).
x_2 : torch.Tensor, shape = (n_faces, in_channels_2)
    Input features on the faces (2-cells).
neighborhood_1_to_1 : torch.Tensor, shape = (n_edges, n_edges)
    Upper-adjacency matrix of rank 1.
neighborhood_2_to_1 : torch.Tensor, shape = (n_edges, n_faces)
    Boundary matrix of rank 2.
neighborhood_0_to_1 : torch.Tensor, shape = (n_edges, n_nodes)
    Coboundary matrix of rank 1.
"""

x_0 = torch.randn(n_nodes, in_channels_0)
x_1 = torch.randn(n_edges, in_channels_1)
x_2 = torch.randn(n_faces, in_channels_2)

neighborhood_1_to_1 = torch.ones(n_edges, n_edges)
neighborhood_2_to_1 = torch.ones(n_edges, n_faces)
neighborhood_0_to_1 = torch.ones(n_edges, n_nodes)

output = model(x_0, x_1, x_2, neighborhood_1_to_1, neighborhood_2_to_1, neighborhood_0_to_1)

print(output.size())
print('Done')
