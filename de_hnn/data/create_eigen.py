import numpy as np
import gzip
import json
from scipy.stats import binned_statistic_2d
import time
import pickle

import torch
import torch.nn.functional as F
from numpy.linalg import eigvals
from torch_geometric.utils import (get_laplacian, to_scipy_sparse_matrix, to_undirected, to_dense_adj)
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_scatter import scatter_add

# Scipy eigendecomposition
from scipy.sparse.linalg import eigsh

abs_dir = '/data/zluo/'
raw_data_dir = abs_dir + 'new_data/'

designs_list = [
    'superblue1',
    'superblue2',
    'superblue3',
    'superblue4',
    'superblue5',
    'superblue6',
    'superblue7',
    'superblue18',
    'superblue19',
    'superblue9',
    'superblue11',
    'superblue14',
    'superblue16'
]
num_designs = len(designs_list)
num_variants_list = [
    5,
    5,
    6,
    5,
    6,
    6,
    6,
    5,
    6,
    6,
    6,
    6,
    6
]

assert num_designs == len(num_variants_list)

# Generate all names
sample_names = []
corresponding_design = []
corresponding_variant = []
for idx in range(num_designs):
    for variant in range(num_variants_list[idx]):
        sample_name = raw_data_dir + designs_list[idx] + '/' + str(variant + 1) + '/'
        sample_names.append(sample_name)
        corresponding_design.append(designs_list[idx])
        corresponding_variant.append(variant + 1)

# Synthetic data
N = len(sample_names)
data_dir = '2023-03-06_data/'

# Eigendecomposition
k = 10
for idx in [50, 56, 62, 68]:#[0, 5, 10, 16, 21, 27, 33, 39, 40, 44]:
    print(idx)
    file_name = data_dir + str(idx) + '.bipartite.pkl'
    f = open(file_name, 'rb')
    dictionary = pickle.load(f)
    f.close()
    
    instance_idx = dictionary['instance_idx']
    net_idx = dictionary['net_idx']

    file_name = data_dir + str(idx) + '.node_features.pkl'
    f = open(file_name, 'rb')
    dictionary = pickle.load(f)
    f.close()

    num_instances = dictionary['num_instances']
    num_nets = dictionary['num_nets']

    # Construct the edge list
    net_idx += num_instances
    v1 = torch.unsqueeze(torch.Tensor(np.concatenate([instance_idx, net_idx], axis = 0)).long(), dim = 1)
    v2 = torch.unsqueeze(torch.Tensor(np.concatenate([net_idx, instance_idx], axis = 0)).long(), dim = 1)

    undir_edge_index = torch.transpose(torch.cat([v1, v2], dim = 1), 0, 1)

    # Create symmetric graph Laplacian
    L = to_scipy_sparse_matrix(
        *get_laplacian(undir_edge_index, normalization = "sym", num_nodes = num_instances + num_nets)
    )

    # Sparse Scipy - Eigendecomposition
    t = time.time()
    evals, evects = eigsh(L, k = k)
    print('Computation time:', time.time() - t)

    dictionary = {
        'evals': evals,
        'evects': evects
    }
    fn = data_dir + str(idx) + '.eigen.' + str(k) + '.pkl'
    f = open(fn, "wb")
    pickle.dump(dictionary, f)
    f.close()
    print('Save file', fn)

print('Done')
