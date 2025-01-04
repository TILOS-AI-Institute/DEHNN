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

# Message Passing Neural Networks (MPNN) with virtual node (VN)
class MPNN_VN(nn.Module):
    def __init__(self, num_layers, node_dim, edge_dim, hidden_dim, z_dim, num_outputs = 1, regression = True, device = 'cuda'):
        super(MPNN_VN, self).__init__()
        self.num_layers = num_layers
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        self.num_outputs = num_outputs
        self.regression = regression
        self.device = device

        self.encoder = GraphEncoder(self.num_layers, self.node_dim, self.edge_dim, self.hidden_dim, self.z_dim, device = device).to(device = device)

        self.fc1 = nn.Linear(self.z_dim, 512).to(device = device)
        self.fc2 = nn.Linear(512, self.num_outputs).to(device = device)

    def forward(self, adj, node_feat, edge_feat = None):
        batch_size = adj.size(0)

        # Call the graph encoder
        latent = self.encoder(adj.float(), node_feat, edge_feat)
        latent = torch.sum(latent, dim = 1)

        # Prediction
        hidden = torch.tanh(self.fc1(latent))
        predict = self.fc2(hidden)

        # If the task is classification then apply softmax
        if self.regression is False:
            predict = torch.softmax(predcit, dim = 1)

        return predict, latent

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
        self.vn_net = nn.ModuleList()
        self.vn_mix = nn.ModuleList()

        for layer in range(self.num_layers):
            self.base_net.append(GraphConvSparse(self.hidden_dim, self.hidden_dim, device = self.device).to(device = device))
            self.vn_net.append(
                    torch.nn.Sequential(
                        torch.nn.Linear(self.hidden_dim, 2 * self.hidden_dim), 
                        torch.nn.BatchNorm1d(2 * self.hidden_dim), 
                        torch.nn.ReLU(),
                        torch.nn.Linear(2 * self.hidden_dim, self.hidden_dim), 
                        torch.nn.BatchNorm1d(self.hidden_dim), 
                        torch.nn.ReLU()
                    ).to(device = device)
            )

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

        # Virtual node
        vn_hidden = torch.mean(node_hidden, dim = 1)

        for layer in range(len(self.base_net)):
            if layer == 0:
                hidden = self.base_net[layer](adj, node_hidden)
            else:
                hidden = self.base_net[layer](adj, hidden)

            # Virtual node
            vn_update = torch.mean(hidden, dim = 1)
            vn_hidden = self.vn_net[layer](vn_update)

            # Broadcast virtual node's signal to other original nodes
            vn_broadcast = torch.cat([vn_hidden.unsqueeze(dim = 1) for node in range(hidden.size(1))], dim = 1)
            
            # Update for other nodes with the virtual node's signal
            hidden = hidden + vn_broadcast
            
            if edge_feat is not None and self.edge_dim is not None:
                hidden = torch.cat((hidden, torch.tanh(torch.einsum('bijc,bjk->bik', edge_hidden, hidden))), dim = 2)
                hidden = torch.tanh(self.combine_net[layer](hidden))
        
            all_hidden.append(hidden)
        
        if self.use_concat_layer == True:
            hidden = torch.cat(all_hidden, dim = 2)

        latent = torch.tanh(self.latent_fc1(hidden))
        latent = torch.tanh(self.latent_fc2(latent))
        return latent

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
        x = self.activation(torch.matmul(adj, x))
        return x

# Glorot initialization
def glorot_init(input_dim, output_dim):
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = torch.rand(input_dim, output_dim) * 2 * init_range - init_range
    return nn.Parameter(initial)

