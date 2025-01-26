import os
import numpy as np
import pickle
import torch
import torch.nn
from sklearn.metrics import accuracy_score, precision_score, recall_score


import torch
import numpy as np
from sklearn.metrics import r2_score as sklearn_r2_score
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score

def compute_accuracy(logits, targets):
    predicted_classes = torch.argmax(logits, dim=1)
    correct_predictions = (predicted_classes.long() == targets.long()).sum().item()
    accuracy = correct_predictions / targets.size(0)
    return accuracy

# Function to compute accuracy, precision, and recall
def compute_metrics(true_labels, predicted_labels):
    # Accuracy
    accuracy = accuracy_score(true_labels, predicted_labels)
    
    # Precision
    precision = precision_score(true_labels, predicted_labels, average='binary')
    
    # Recall
    recall = recall_score(true_labels, predicted_labels, average='binary')
    
    return accuracy, precision, recall

def compute_r_squared(y_pred, y_true):
    """
    Compute R-squared for a regression model.
    
    Parameters:
        y_true (torch.Tensor): Ground truth (actual values).
        y_pred (torch.Tensor): Model predictions.

    Returns:
        float: R-squared value.
    """
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()
    ss_total = torch.sum((y_true - torch.mean(y_true)) ** 2)
    ss_residual = torch.sum((y_true - y_pred) ** 2)
    r2 = 1 - (ss_residual / ss_total)
    return r2.item()

def mape(y_pred, y_true, epsilon=0.01):
    """
    Computes Mean Absolute Percentage Error (MAPE).
    
    Args:
        y_true (torch.Tensor): Ground truth values.
        y_pred (torch.Tensor): Predicted values.
        epsilon (float): Small value to avoid division by zero.
    
    Returns:
        torch.Tensor: MAPE value.
    """
    percentage_error = torch.abs((y_true - y_pred) / (y_true + epsilon))
    mape_value = torch.mean(percentage_error) * 100
    return mape_value

def weighted_mse_loss(input, target, weight):
    return torch.mean(weight * (input - target) ** 2)

def convert_to_util_map(pos_lst, node_features, num_sites_x, num_sites_y, device, max_sum):
    """
    Convert the output from dehnn model to utilization map
    """
    node_features = node_features.unsqueeze(1)
    num_channels = node_features.shape[1]
    min_value = torch.min(node_features).item()

    indices = (pos_lst[:, 0].long() * num_sites_y + pos_lst[:, 1].long()).to(device)
    indices = indices.unsqueeze(1).expand(node_features.shape[0], num_channels)

    if max_sum == 'amax':
        util_map = torch.full((num_sites_x * num_sites_y, num_channels), min_value, device=device)
    elif max_sum == 'sum':
        util_map = torch.full((num_sites_x * num_sites_y, num_channels), float(0.), device=device)

    util_map.scatter_reduce_(0, indices, node_features, reduce=max_sum, include_self=True)
    util_map = util_map.view(num_sites_x, num_sites_y, num_channels)

    return util_map

def nrmse(y_true, y_pred):
    mse = torch.mean((y_true - y_pred) ** 2)
    rmse = torch.sqrt(mse)
    norm = y_true.max() - y_true.min()
    return rmse / norm

def ssim(y_true, y_pred, C1=1e-4, C2=9e-4):
    mu_true = torch.mean(y_true)
    mu_pred = torch.mean(y_pred)
    sigma_true = torch.var(y_true)
    sigma_pred = torch.var(y_pred)
    covariance = torch.mean((y_true - mu_true) * (y_pred - mu_pred))
    
    numerator = (2 * mu_true * mu_pred + C1) * (2 * covariance + C2)
    denominator = (mu_true ** 2 + mu_pred ** 2 + C1) * (sigma_true + sigma_pred + C2)
    return numerator / denominator

def mape(y_true, y_pred):
    epsilon = 1e-1  # To avoid division by zero
    absolute_percentage_errors = torch.abs((y_true - y_pred) / (y_true + epsilon))
    return torch.mean(absolute_percentage_errors) * 100

def r2_score(y_true, y_pred):
    ss_total = torch.sum((y_true - torch.mean(y_true)) ** 2)
    ss_residual = torch.sum((y_true - y_pred) ** 2)
    return 1 - (ss_residual / ss_total)

# Function to compute accuracy, precision, and recall
def compute_metrics(true_labels, predicted_labels):
    # Accuracy
    accuracy = accuracy_score(true_labels, predicted_labels)
    
    # Precision
    precision = precision_score(true_labels, predicted_labels, average='binary')
    
    # Recall
    recall = recall_score(true_labels, predicted_labels, average='binary')
    
    return accuracy, precision, recall

class UtilMapConverter:
    def __init__(self, pos_lst, num_sites_x, num_sites_y, device):
        self.shape = (num_sites_x, num_sites_y)
        self.device = device
        self.flat_size = num_sites_x * num_sites_y
        self.indices = (pos_lst[:, 0].long() * num_sites_y + pos_lst[:, 1].long()).to(device)
        self._expanded_indices = None

    def to_util_map(self, node_features, max_sum):
        num_channels = node_features.shape[1]
        
        # Get or create expanded indices
        if self._expanded_indices is None or self._expanded_indices.shape[1] != num_channels:
            self._expanded_indices = self.indices.unsqueeze(1).expand(-1, num_channels)
            
        if max_sum == 'sum':
            util_map = torch.zeros(self.flat_size, num_channels, device=self.device)
            util_map.scatter_add_(0, self._expanded_indices, node_features)
        elif max_sum == 'amax':
            min_value = torch.min(node_features).item()
            util_map = torch.full((self.flat_size, num_channels), min_value, device=self.device)
            util_map.scatter_reduce_(0, self._expanded_indices, node_features, reduce='amax')
            
        return util_map.view(*self.shape, num_channels)

    def to_node_features(self, util_map_flat):
        return util_map_flat[self.indices]

def train_epoch(model, h_dataset, indices, optimizer, criterion_node, criterion_vn_node, criterion_net, device, unet_only):
    loss_node_all = loss_vn_all = loss_net_all = 0
    all_train_idx = 0

    for data_idx in indices:
        data = h_dataset[data_idx//25]
        pos_lst, pos_lst_net, edge_attr, target_node, target_net = data.variant_data_lst[data_idx%25]
        
        optimizer.zero_grad()
        data['node'].x = torch.concat([data['node'].node_features, pos_lst], dim=1)
        data['net'].x = torch.concat([data['net'].net_features, pos_lst_net], dim=1)
        data['node', 'as_a_sink_of', 'net'].edge_attr = edge_attr
        data.pos_lst = pos_lst
        data.num_sites_x = 206
        data.num_sites_y = 300

        node_representation, net_representation, vn_representation = model(data, device)
        
        if net_representation is None:
            net_representation = target_net.to(device)
            
        target_map = convert_to_util_map(pos_lst, target_node.to(device), 206, 300, device, "amax").flatten()
        loss_node = criterion_node(node_representation.flatten(), target_node.to(device))
        loss_net = criterion_net(net_representation.flatten(), target_net.to(device))
        loss_vn = criterion_vn_node(vn_representation.flatten(), target_map.to(device))
        loss = 10*loss_node + 0.001*loss_net + 5*loss_vn

        loss.backward()
        optimizer.step()
        
        loss_node_all += loss_node.item()
        loss_vn_all += loss_vn.item()
        loss_net_all += loss_net.item()
        all_train_idx += 1

    return {
        "loss_node": loss_node_all/all_train_idx,
        "loss_vn": loss_vn_all/all_train_idx,
        "loss_net": loss_net_all/all_train_idx
    }

def validate_epoch(model, h_dataset, indices, criterion_node, criterion_vn_node, criterion_net, device, unet_only):
    loss_node_all = loss_vn_all = loss_net_all = 0
    all_val_idx = 0

    for data_idx in indices:
        data = h_dataset[data_idx//25]
        pos_lst, pos_lst_net, edge_attr, target_node, target_net = data.variant_data_lst[data_idx%25]
        
        data['node'].x = torch.concat([data['node'].node_features, pos_lst], dim=1)
        data['net'].x = torch.concat([data['net'].net_features, pos_lst_net], dim=1)
        data['node', 'as_a_sink_of', 'net'].edge_attr = edge_attr
        data.pos_lst = pos_lst
        data.num_sites_x = 206
        data.num_sites_y = 300

        node_representation, net_representation, vn_representation = model(data, device)
        
        if net_representation is None:
            net_representation = target_net.to(device)
            
        target_map = convert_to_util_map(pos_lst, target_node.to(device), 206, 300, device, "amax").flatten()
        loss_node = criterion_node(node_representation.flatten(), target_node.to(device))
        loss_net = criterion_net(net_representation.flatten(), target_net.to(device))
        loss_vn = criterion_vn_node(vn_representation.flatten(), target_map.to(device))

        loss_node_all += loss_node.item()
        loss_vn_all += loss_vn.item()
        loss_net_all += loss_net.item()
        all_val_idx += 1

    return {
        "loss_node": loss_node_all/all_val_idx,
        "loss_vn": loss_vn_all/all_val_idx,
        "loss_net": loss_net_all/all_val_idx
    }