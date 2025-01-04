import torch
from hnhn import HNHN

in_channels = 10
hidden_channels = 20
n_layers = 3

num_instances = 1000
num_nets = 100

instance_features = torch.randn(num_instances, in_channels)
adj = torch.randn(num_instances, num_nets).to_sparse()

model = HNHN(
        in_channels = in_channels, 
        hidden_channels = hidden_channels, 
        incidence_1 = adj,
        n_layers = n_layers
)

instance_predict, net_predict = model(instance_features)

print("Instance predict:", instance_predict.size())
print("Net predict:", net_predict.size())
print("Done")
