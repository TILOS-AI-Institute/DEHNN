import torch
from hmpnn import HMPNN

in_channels = 10
hidden_channels = 20
n_layers = 2

model = HMPNN(in_channels = in_channels, hidden_channels = hidden_channels, n_layers = n_layers)

num_instances = 1000
num_nets = 100

instance_features = torch.randn(num_instances, in_channels)
net_features = torch.randn(num_nets, in_channels)
adj = torch.randn(num_instances, num_nets).to_sparse()

instance_predict, net_predict = model(instance_features, net_features, adj)

print("Instance predict:", instance_predict.size())
print("Net predict:", net_predict.size())
print('Done')
