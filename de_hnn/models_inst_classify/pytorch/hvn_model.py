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

# Hierarchy of virtual nodes (HVN)
class HVN(nn.Module):
    def __init__(self, clusters, num_layers, node_dim, edge_dim, hidden_dim, z_dim, num_outputs = 1, regression = True, device = 'cuda'):
        super(HVN, self).__init__()
        self.clusters = clusters
        self.num_layers = num_layers
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        self.num_outputs = num_outputs
        self.regression = regression
        self.device = device

        # First graph encoder
        self.first_encoder = GraphEncoder(self.num_layers, self.node_dim, self.edge_dim, self.hidden_dim, self.z_dim, device = device).to(device = device)

        # For learning to clusters
        self.clustering_modules = nn.ModuleList()
        l = len(self.clusters) - 1
        while l >= 0:
            size = self.clusters[l]
            self.clustering_modules.append(nn.Linear(self.z_dim, size).to(device = device))
            l -= 1

        # Second graph encoder
        # Option 1: Use the latents from the first encoder
        # self.second_encoder = GraphEncoder(self.num_layers, self.z_dim, None, self.hidden_dim, self.z_dim, device = device).to(device = device)
        # Option 2: Use the input node features
        self.second_encoder = GraphEncoder(self.num_layers, self.node_dim, None, self.hidden_dim, self.z_dim, device = device).to(device = device)

        # Top layers
        self.fc1 = nn.Linear(2 * self.z_dim, 512).to(device = device)
        self.fc2 = nn.Linear(512, self.num_outputs).to(device = device)

    def forward(self, adj, node_feat, edge_feat = None):
        batch_size = adj.size(0)

        # Call the first graph encoder
        first_latent = self.first_encoder(adj.float(), node_feat, edge_feat)
        
        # Extend the adjacency matrix by learning to cluster
        A = adj.float()
        latent = first_latent
        all_latents = [first_latent]
        N = latent.size(1)

        for i in range(len(self.clustering_modules)):
            # Assignment score
            assign_score = self.clustering_modules[i](latent)
            
            # Gumbel softmax (hard assignment)
            assign_matrix = F.gumbel_softmax(assign_score, tau = 1, hard = True, dim = 2)

            # Shrinked latent
            shrinked_latent = torch.matmul(assign_matrix.transpose(1, 2), latent)

            # Latent normalization
            latent = F.normalize(shrinked_latent, dim = 1)

            # Extend the current adjacency matrix
            size = assign_matrix.size(2)
            if i > 0:
                assign_matrix = torch.cat([torch.zeros(batch_size, N - assign_matrix.size(1), size).to(device = self.device), assign_matrix], dim = 1)
            N += size
            A = torch.cat([A, assign_matrix], dim = 2)
            bottom_rows = torch.cat([torch.transpose(assign_matrix, 1, 2), torch.zeros(batch_size, size, size).to(device = self.device)], dim = 2)
            A = torch.cat([A, bottom_rows], dim = 1)

            # Extend the latent
            all_latents.append(latent)

        # Call the second graph encoder
        all_latents = torch.cat(all_latents, dim = 1)

        # Option 1: Use latents from the first encoder
        # second_latent = self.second_encoder(A, all_latents, None)

        # Option 2: Use the input node features 
        second_latent = self.second_encoder(A, torch.cat([node_feat, torch.zeros(batch_size, A.size(1) - node_feat.size(1), node_feat.size(2)).to(device = self.device)], dim = 1), None)

        # Readout for the first latent
        first_latent = torch.sum(first_latent, dim = 1)

        # Readout for the second latent
        second_latent = torch.sum(second_latent, dim = 1)

        # Representation
        rep = torch.cat([first_latent, second_latent], dim = 1)

        # Prediction
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
            self.edge_fc1 = nn.Linear(self.edge_dim, 128).to(device = device)
            self.edge_fc2 = nn.Linear(128, 1).to(device = device)

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
                hidden = torch.cat((hidden, torch.tanh(torch.einsum('bijk,bjc->bic', edge_hidden, hidden))), dim = 2)
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
        self.base_net = nn.ModuleList()
        
        # Option 2: Fixed clustering
        # self.base_net = []

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

