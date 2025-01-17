import os
import numpy as np
import pickle
import torch
import torch.nn
from sklearn.metrics import accuracy_score, precision_score, recall_score


import torch
import numpy as np
from sklearn.metrics import r2_score as sklearn_r2_score
from skimage.metrics import structural_similarity as sk_ssim
import torch.nn.functional as F

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

def nrmse(target, pred):
    """
    Normalized Root Mean Square Error
    """
    mse = torch.mean((target - pred) ** 2)
    rmse = torch.sqrt(mse)
    return rmse / (torch.max(target) - torch.min(target))

def ssim(target, pred, data_range=None):
    """
    Structural Similarity Index
    """
    if torch.is_tensor(target):
        target = target.detach().cpu().numpy()
    if torch.is_tensor(pred):
        pred = pred.detach().cpu().numpy()
        
    if data_range is None:
        data_range = max(target.max() - target.min(), pred.max() - pred.min())
    
    return torch.tensor(sk_ssim(target, pred, data_range=data_range))

def mape(target, pred, epsilon=1e-10):
    """
    Mean Absolute Percentage Error
    """
    return torch.mean(torch.abs((target - pred) / (target + epsilon))) * 100

def r2_score(target, pred):
    """
    R-squared score
    """
    if torch.is_tensor(target):
        target = target.cpu().numpy()
    if torch.is_tensor(pred):
        pred = pred.cpu().numpy()
        
    return torch.tensor(sklearn_r2_score(target.flatten(), pred.flatten()))