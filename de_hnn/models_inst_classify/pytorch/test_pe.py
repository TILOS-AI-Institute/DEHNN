import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, Adagrad
from torch import optim
from torch.utils.data import DataLoader
import numpy as np
import os
import time

from position_encoding import LapPE

# Test 1
A = [
        [0, 1, 0, 0, 1, 0],
        [1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 0],
        [0, 0, 1, 0, 1, 1],
        [1, 1, 0, 1, 0, 0],
        [0, 0, 0, 1, 0, 0]
    ]
A = torch.from_numpy(np.array(A)).type(torch.FloatTensor)
A = A.unsqueeze(dim = 0)

pe = LapPE(num_vectors = 2, norm_laplacian = False)
print(pe.get_laplacian(A))

position = pe(A)
print(position)

# Test 2
A = [
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0]
    ]
A = torch.from_numpy(np.array(A)).type(torch.FloatTensor)
A = A.unsqueeze(dim = 0)

pe = LapPE(num_vectors = 2, norm_laplacian = True)
print(pe.get_laplacian(A))

position = pe(A)
print(position)

print('Done')
