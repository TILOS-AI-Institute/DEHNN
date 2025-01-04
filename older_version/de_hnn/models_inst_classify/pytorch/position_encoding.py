import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, Adagrad
from torch import optim
from torch.utils.data import DataLoader
import numpy as np
import os
import time

# Graph Laplacian Positional Encoding
class LapPE(nn.Module):
    def __init__(self, num_vectors, norm_laplacian = False, device = 'cpu'):
        super(LapPE, self).__init__()
        # Number of eigenvectors
        self.num_vectors = num_vectors
        assert(self.num_vectors >= 1)
        # Normalize graph Laplacian or not
        self.norm_laplacian = norm_laplacian
        # Device 'cuda' or 'cpu'
        self.device = device

    def get_laplacian(self, A, norm_laplacian = True):
        # Batch size
        B = A.size(0)
        # Number of nodes
        N = A.size(1)
        # Create B identity matrices
        e = torch.eye(N)
        e = e.reshape((1, N, N))
        eye = e.repeat(B, 1, 1).to(device = self.device)
        # Compute the diagonal matrix of node degree
        d = torch.sum(A, dim = 2)
        # Computation of normalized graph Laplacian or un-normalized
        if self.norm_laplacian == True:
            d[d == 0] = 1
            d = 1.0 / torch.sqrt(d)
            D = torch.einsum('bij,bi->bij', (eye, d))
            L = eye - torch.matmul(torch.matmul(D, A), D)
        else:
            D = torch.einsum('bij,bi->bij', (eye, d))
            L = D - A
        return L

    def forward(self, A):
        L = self.get_laplacian(A, norm_laplacian = self.norm_laplacian)
        values, vectors = torch.linalg.eig(L)
        real = torch.real(vectors)
        return real[:, :self.num_vectors]
