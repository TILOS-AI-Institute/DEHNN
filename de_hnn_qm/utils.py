import torch
import numpy as np
import networkx as nx
from torch_geometric.utils import get_embeddings

def walk_to_edge_set(walk):
    edges = set()
    for i in range(len(walk) - 1):
        edges.add((walk[i], walk[i+1]))
    return edges

def cg_walks_to_edge_set(walks):
    edges = set()
    for walk in walks:
        for i in range(len(walk) - 1):
            edges.add((walk[i][1], walk[i+1][1]))
    return edges

def edge_set_to_edge_mask(edges, edge_index):
    temp = torch.zeros(edge_index.shape[1])
    for i in range(edge_index.shape[1]):
        if (int(edge_index[1,i]), int(edge_index[0,i])) in edges:
            # or (int(edge_index[0,i]), int(edge_index[1,i])) in edges:
            temp[i] = 1
    return temp