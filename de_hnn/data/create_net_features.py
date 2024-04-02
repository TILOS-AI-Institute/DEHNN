import torch
import torch.nn as nn

import math
import pickle
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from tqdm import tqdm

import sys
sys.path.insert(1, '../experiments/single_design/')
from pyg_dataset import pyg_dataset as pyg
from pyg_dataset_net import pyg_dataset as pyg_net
from pyg_dataset_sparse import pyg_dataset as pyg_sparse
import os

os.chdir('../experiments/single_design/')

for graph_index in tqdm(range(50, 74)):

    dataset = pyg_sparse('../../data/2023-03-06_data/', graph_index = graph_index, target = 'demand', load_pe=True, num_eigen=10, load_global_info = False, load_pd=True, vn=False, net=True)
    data = dataset[0]

    x = data.x
    net_inst_adj = data.net_inst_adj
    x_net = data.x_net

    net_agg = torch.mm(net_inst_adj, x)

    net_all = torch.cat([net_agg, x_net], dim=1)

    dictionary = {'instance_features': net_all.numpy()}

    f = open('../../data/2023-03-06_data/' + str(graph_index) + '.net_features.pkl', 'wb')
    pickle.dump(dictionary, f)
    f.close()