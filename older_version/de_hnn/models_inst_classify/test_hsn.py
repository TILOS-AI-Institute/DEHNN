import torch

from hsn import HSN

in_channels = 10
n_layers = 3

model = HSN(channels = in_channels, n_layers = n_layers)

num_instances = 1000
num_nets = 100

# Instance input features
instance_features = torch.randn(num_instances, in_channels)

# Instance to net adj
instance_2_net = torch.randn(num_instances, num_nets)

# Instance to instance adj
instance_2_instance = torch.randn(num_instances, num_instances)

output = model(instance_features, instance_2_net, instance_2_instance)

print(output.size())

print('Done')
