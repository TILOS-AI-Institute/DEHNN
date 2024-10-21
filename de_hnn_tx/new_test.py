import os
import numpy as np
import pickle
import torch
import torch.nn
from torch_geometric.data import Dataset
from torch_geometric.data import Data, HeteroData
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from utils import *
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
sys.path.insert(1, 'data/')
from pyg_dataset import NetlistDataset
from data.utils import compute_degrees

sys.path.append("models/layers/")
from models.model import GNN_node

from sklearn.metrics import accuracy_score, precision_score, recall_score

# Function to compute accuracy, precision, and recall
def compute_metrics(true_labels, predicted_labels):
    # Accuracy
    accuracy = accuracy_score(true_labels, predicted_labels)
    
    # Precision
    precision = precision_score(true_labels, predicted_labels, average='binary')
    
    # Recall
    recall = recall_score(true_labels, predicted_labels, average='binary')
    
    return accuracy, precision, recall

h_dataset = torch.load("147_dataset.pt")
for data in h_dataset:
    data['node'].x = torch.concat([data['node'].x, data.pos + (torch.rand(1) * 100)], dim=1)

device = "cuda"
model = torch.load("dehnn_model_single.pt").to(device)

criterion_node = nn.CrossEntropyLoss()
criterion_net = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
best_total_val = None
all_indices = [idx for idx in range(len(h_dataset))]
all_train_indices, all_valid_indices, all_test_indices = all_indices[:8], all_indices[8:], all_indices[8:]

np.random.shuffle(all_train_indices)
loss_node_all = 0
loss_net_all = 0
val_loss_node_all = 0
val_loss_net_all = 0

precision_all = 0
recall_all = 0

model.train()
all_train_idx = 0
for data_idx in tqdm(all_train_indices):
    data = h_dataset[data_idx]
    optimizer.zero_grad()
    data['node'].x = torch.concat([data['node'].x[:, :-2], data.pos + (torch.rand(1) * 100)], dim=1)
    node_representation, net_representation = model(data, device)
    
    target_node = data['node'].y.long()
    target_net = data['net'].y.float()
        
    bias_distribution = torch.tensor([torch.mean(target_node.float()).item() * 10, 1.0])
    keep_probabilities = bias_distribution[target_node]
    random_values = torch.rand(len(target_node))
    keep_mask = random_values < keep_probabilities
    
    loss_node = criterion_node(node_representation[keep_mask], target_node[keep_mask].to(device))
    loss_net = criterion_net(net_representation.flatten(), target_net.to(device))
    print(loss_node, loss_net, data_idx)
    loss = loss_node + 0.001*loss_net
    pred_class = torch.argmax(node_representation, dim=1)
    accuracy, precision, recall = compute_metrics(target_node.cpu(), pred_class.detach().cpu())
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")

    pos_lst = data.pos
    x_lst = pos_lst[:, 0].cpu().detach().flatten()
    y_lst = pos_lst[:, 1].cpu().detach().flatten()
    fig, axs = plt.subplots(1, 2, figsize=(5, 2.5))
    axs[0].scatter(x_lst, y_lst, c=target_node.cpu(), s=1)
    axs[1].scatter(x_lst, y_lst, c=pred_class.detach().cpu(), s=1)
    plt.savefig(f'{data_idx}.png')