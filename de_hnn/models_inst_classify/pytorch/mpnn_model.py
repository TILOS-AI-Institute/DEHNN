import torch
import torch.nn as nn
from torch.nn import Sequential as Seq, Linear, ReLU
import torch.nn.functional as F
from torch.optim import Adam, Adagrad
from torch import optim
from torch.utils.data import DataLoader
import numpy as np
import scipy
import os
import time

# Message Passing Neural Networks (MPNN)
class MPNN(nn.Module):
    def __init__(self, num_layers, node_dim, edge_dim, hidden_dim, z_dim, num_outputs = 1, regression = True, device = 'cuda'):
        super(MPNN, self).__init__()
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

    def forward(self, adj, adj_d, adj_s, node_feat, edge_feat = None):
        batch_size = adj.size(0)

        # Call the graph encoder
        latent = self.encoder(adj, adj_d, adj_s, node_feat, edge_feat)

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

    def forward(self, adj, adj_d, adj_s, node_feat, edge_feat = None):
        node_hidden = torch.tanh(self.node_fc1(node_feat))
        node_hidden = torch.tanh(self.node_fc2(node_hidden))
        
        if edge_feat is not None and self.edge_dim is not None:
            edge_hidden = torch.tanh(self.edge_fc1(edge_feat))
            edge_hidden = torch.tanh(self.edge_fc2(edge_hidden))

        all_hidden = [node_hidden]
        for layer in range(len(self.base_net)):
            if layer == 0:
                hidden = self.base_net[layer](adj, adj_d, adj_s, node_hidden)
            else:
                hidden = self.base_net[layer](adj, adj_d, adj_s, hidden)
            
            if edge_feat is not None and self.edge_dim is not None:
                hidden = torch.cat((hidden, torch.tanh(torch.einsum('bijc,bjk->bik', edge_hidden, hidden))), dim = 2)
                hidden = torch.tanh(self.combine_net[layer](hidden))
        
            all_hidden.append(hidden)
        
        if self.use_concat_layer == True:
            hidden = torch.cat(all_hidden, dim = 1)
        
        latent = torch.tanh(self.latent_fc1(hidden))
        latent = torch.tanh(self.latent_fc2(latent))
        return latent

# Graph convolution module
class GraphConvSparse(nn.Module):
    def __init__(self, input_dim, output_dim, device = 'cuda', **kwargs):
        super(GraphConvSparse, self).__init__(**kwargs)
        self.phi = Seq(Linear(input_dim, output_dim),
                       ReLU(),
                       Linear(output_dim, output_dim))
        
        self.psi1 = Seq(Linear(output_dim, output_dim),
                       ReLU(),
                       Linear(output_dim, output_dim))
        
        self.psi2 = Seq(Linear(output_dim, output_dim),
                       ReLU(),
                       Linear(output_dim, output_dim))

        self.mlp = Seq(Linear(output_dim * 3, output_dim * 3),
                       ReLU(),
                       Linear(output_dim * 3, output_dim))

    def forward(self, net_inst_adj, inst_net_adj_v_drive, inst_net_adj_v_sink, x):
        
        h = self.phi(x)
        
        net_agg = torch.mm(net_inst_adj, h)
        
        h_drive = self.psi1(torch.mm(inst_net_adj_v_drive, net_agg))
        
        h_sink = self.psi2(torch.mm(inst_net_adj_v_sink, net_agg))
        
        return self.mlp(torch.concat([x, h_drive, h_sink], dim=1))

# Glorot initialization
def glorot_init(input_dim, output_dim):
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = torch.rand(input_dim, output_dim) * 2 * init_range - init_range
    return nn.Parameter(initial)

