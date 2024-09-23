import torch

def compute_degrees(edge_index, num_nodes=None):
    # If num_nodes is not provided, infer it from edge_index
    if num_nodes is None:
        num_nodes = edge_index.max().item() + 1
    
    # Create a degree tensor initialized to zero
    degree = torch.zeros(num_nodes, dtype=torch.long)
    
    # Count the number of edges connected to each node
    degree.scatter_add_(0, edge_index[0], torch.ones(edge_index.size(1), dtype=torch.long))
    
    return degree
