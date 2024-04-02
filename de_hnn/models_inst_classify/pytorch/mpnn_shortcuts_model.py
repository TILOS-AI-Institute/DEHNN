import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, Adagrad
from torch import optim
from torch.utils.data import DataLoader
import numpy as np
import scipy
import os
import time

# Message Passing Neural Networks (MPNN) with learning to add shortcuts
class MPNN_Shortcuts(nn.Module):
    def __init__(self, num_layers, node_dim, edge_dim, hidden_dim, z_dim, num_outputs = 1, regression = True, distance_threshold = 10, device = 'cuda'):
        super(MPNN_Shortcuts, self).__init__()
        self.distance_threshold = distance_threshold
        self.num_layers = num_layers
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        self.num_outputs = num_outputs
        self.regression = regression
        self.device = device

        self.first_encoder = GraphEncoder(self.num_layers, self.node_dim, self.edge_dim, self.hidden_dim, self.z_dim, device = device).to(device = device)
        self.second_encoder = GraphEncoder(self.num_layers, self.node_dim, self.edge_dim, self.hidden_dim, self.z_dim, device = device).to(device = device)

        D = self.z_dim * 2
        self.fc1 = nn.Linear(D, 512).to(device = device)
        self.fc2 = nn.Linear(512, self.num_outputs).to(device = device)

    def forward(self, adj, node_feat, edge_feat = None):
        # Find all the shortest paths and filter out all the short paths that are less than or equal to the distance threshold
        batch_size = adj.size(0)
        paths = []
        paths_inverse = []

        for b in range(batch_size):
            # The adjacency
            A = adj[b].detach().cpu().numpy()

            # All-pair shortest paths
            P = scipy.sparse.csgraph.shortest_path(A, directed = False, unweighted = True)
            P = torch.from_numpy(P).unsqueeze(dim = 0)

            # Remove all Infinity
            P[P == float("Inf")] = 1

            # Compute 1/distance^2 where distance is the shortest path distance
            P_inverse = 1.0 / (P * P)
            P_inverse[P_inverse == float("Inf")] = 0

            # Filter out all short distances
            P[P <= self.distance_threshold] = 0
            P[P > self.distance_threshold] = 1
            P_inverse[P == 0] = 0
            
            paths.append(P)
            paths_inverse.append(P_inverse)
        
        paths = torch.cat(paths, dim = 0).to(device = self.device)
        paths_inverse = torch.cat(paths_inverse, dim = 0).to(device = self.device)

        # First MPNN
        first_latent = self.first_encoder(adj, node_feat, edge_feat)
        
        # Create a score for each pair
        square_map = torch.sigmoid(torch.matmul(first_latent, first_latent.transpose(1, 2)))
        
        # Scale the probabilities with 1/d^2 where d is the shortest path distance
        square_map = square_map * paths_inverse
        square_map = square_map.unsqueeze(dim = 3)

        # Create a tensor for edge/non-edge probabilities
        edge_nonedge_probability = torch.cat((square_map, 1.0 - square_map), dim = 3)

        # Gumbel-max to sample shortcuts
        edge_sample = F.gumbel_softmax(edge_nonedge_probability, tau = 1, hard = True, dim = 3)
        edge_sample = edge_sample[:, :, :, 0]
        
        # Filter out all shortcuts with short distances (only keep long distances)
        shortcuts = edge_sample * paths

        # The new adjacency with newly added shortcuts
        adj = adj + shortcuts

        # Second MPNN on the new adjacency with the shortcuts
        second_latent = self.second_encoder(adj.float(), node_feat, edge_feat)

        # Readout for first MPNN
        first_rep = torch.mean(first_latent, dim = 1)

        # Readout for second MPNN
        second_rep = torch.mean(second_latent, dim = 1)

        # Scalar prediction
        rep = torch.cat([first_rep, second_rep], dim = 1)
        hidden = torch.tanh(self.fc1(rep))
        predict = self.fc2(hidden)

        # If the task is classification then apply softmax
        if self.regression is False:
            predict = torch.softmax(predcit, dim = 1)

        return predict, rep

# Graph encoder module
class GraphEncoder(nn.Module):
    def __init__(self, num_layers, node_dim, edge_dim, hidden_dim, z_dim, use_concat_layer = True, device = 'cuda', **kwargs):
        super(GraphEncoder, self).__init__(**kwargs)
        self.num_layers = num_layers
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        self.use_concat_layer = use_concat_layer
        self.device = device
        
        self.node_fc1 = nn.Linear(self.node_dim, 256).to(device = device)
        self.node_fc2 = nn.Linear(256, self.hidden_dim).to(device = device)
        
        if self.edge_dim is not None:
            self.edge_fc1 = nn.Linear(self.edge_dim, 256).to(device = device)
            self.edge_fc2 = nn.Linear(256, self.hidden_dim).to(device = device)

        self.base_net = nn.ModuleList()
        self.combine_net = nn.ModuleList()
        for layer in range(self.num_layers):
            self.base_net.append(GraphConvSparse(self.hidden_dim, self.hidden_dim, device = self.device).to(device = device))
            if self.edge_dim is not None:
                self.combine_net.append(nn.Linear(2 * self.hidden_dim, self.hidden_dim).to(device = device))

        if self.use_concat_layer == True:
            self.latent_fc1 = nn.Linear((self.num_layers + 1) * self.hidden_dim, 256).to(device = device)
            self.latent_fc2 = nn.Linear(256, self.z_dim).to(device = device)
        else:
            self.latent_fc1 = nn.Linear(self.hidden_dim, 256).to(device = device)
            self.latent_fc2 = nn.Linear(256, self.z_dim).to(device = device)

    def forward(self, adj, node_feat, edge_feat = None):
        node_hidden = torch.tanh(self.node_fc1(node_feat))
        node_hidden = torch.tanh(self.node_fc2(node_hidden))
        
        if edge_feat is not None and self.edge_dim is not None:
            edge_hidden = torch.tanh(self.edge_fc1(edge_feat))
            edge_hidden = torch.tanh(self.edge_fc2(edge_hidden))

        all_hidden = [node_hidden]
        for layer in range(len(self.base_net)):
            if layer == 0:
                hidden = self.base_net[layer](adj, node_hidden)
            else:
                hidden = self.base_net[layer](adj, hidden)
            
            if edge_feat is not None and self.edge_dim is not None:
                hidden = torch.cat((hidden, torch.tanh(torch.einsum('bijc,bjk->bik', edge_hidden, hidden))), dim = 2)
                hidden = torch.tanh(self.combine_net[layer](hidden))
        
            all_hidden.append(hidden)
        
        if self.use_concat_layer == True:
            hidden = torch.cat(all_hidden, dim = 2)

        latent = torch.tanh(self.latent_fc1(hidden))
        latent = torch.tanh(self.latent_fc2(latent))
        return latent

# Graph clustering module
class GraphCluster(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, z_dim, device = 'cuda', **kwargs):
        super(GraphCluster, self).__init__(**kwargs)
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        self.device = device

        self.fc1 = nn.Linear(self.input_dim, 128).to(device = device)
        self.fc2 = nn.Linear(128, self.hidden_dim).to(device = device)

        # Option 1: Learnable clustering
        # self.base_net = nn.ModuleList()
        
        # Option 2: Fixed clustering
        self.base_net = []

        for layer in range(self.num_layers):
            self.base_net.append(GraphConvSparse(self.hidden_dim, self.hidden_dim, device = self.device).to(device = device))

        self.assign_net = GraphConvSparse(self.hidden_dim, self.z_dim, device = self.device).to(device = device)

    def forward(self, adj, X):
        hidden = torch.sigmoid(self.fc1(X))
        hidden = torch.sigmoid(self.fc2(hidden))
        for net in self.base_net:
            hidden = net(adj, hidden)
        assign = self.assign_net(adj, hidden)
        return assign

# Graph convolution module
class GraphConvSparse(nn.Module):
    def __init__(self, input_dim, output_dim, activation = torch.tanh, device = 'cuda', **kwargs):
        super(GraphConvSparse, self).__init__(**kwargs)
        self.weight = glorot_init(input_dim, output_dim)
        self.activation = activation
        self.device = device

    def forward(self, adj, inputs):
        # If we use the adjacency matrix instead of the graph Laplacian
        B = adj.size(0)
        N = adj.size(1)
        e = torch.eye(N)
        e = e.reshape((1, N, N))
        eye = e.repeat(B, 1, 1).to(device = self.device)
        D = 1.0 / torch.sum(adj + eye, dim = 2)
        adj = torch.einsum('bi,bij->bij', (D, adj))

        x = inputs
        x = torch.matmul(x, self.weight)
        x = torch.matmul(adj, x)
        outputs = self.activation(x)
        return outputs

# Glorot initialization
def glorot_init(input_dim, output_dim):
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = torch.rand(input_dim, output_dim) * 2 * init_range - init_range
    return nn.Parameter(initial)

