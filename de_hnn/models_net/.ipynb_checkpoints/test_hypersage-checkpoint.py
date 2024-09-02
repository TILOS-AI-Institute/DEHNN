import torch
from hypersage import HyperSAGE

in_channels = 10
hidden_channels = 20
n_layers = 3

model = HyperSAGE(in_channels = in_channels, hidden_channels = hidden_channels, n_layers = n_layers)

num_instances = 100
num_nets = 100

instance_features = torch.randn(num_instances, in_channels)
adj = torch.randn(num_instances, num_nets).to_sparse()

output = model(instance_features, adj)
print(output.size())

print('Done')
