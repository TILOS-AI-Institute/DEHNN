import numpy as np
import os
import time
import scipy
import pickle
import networkx as nx
from tqdm import tqdm

# Connection data 
abs_dir = '/data/zluo/'
raw_data_dir = abs_dir + 'new_data/'
data_dir = '2023-03-06_data'

for sample in tqdm(range(50, 74)):

    graph_rep = 'node_features'
    fn = data_dir + '/' + str(sample) + f'.{graph_rep}.pkl'
    f = open(fn, "rb")
    d = pickle.load(f)
    f.close()
    print('Read file', fn)

    # Connection data 
    graph_rep = 'bipartite'
    fn = data_dir + '/' + str(sample) + f'.{graph_rep}.pkl'
    f = open(fn, "rb")
    dictionary = pickle.load(f)
    f.close()
    print('Read file', fn)
    
    cell = dictionary['instance_idx']
    net = dictionary['net_idx']
    num_instances = d['num_instances']
    num_nets = d['num_nets']
    edge_index = np.array([cell, net + num_instances])
    
    G = nx.Graph(edge_index.T.tolist())
    
    cell_degrees = []
    net_degrees = []
    for node in range(num_instances):
        try:
            cell_degrees.append(G.degree[node])
        except:
            cell_degrees.append(0)

    for node in range(num_instances, num_nets + num_instances):
        try:
            net_degrees.append(G.degree[node])
        except:
            net_degrees.append(0)     
    d = {
        'cell_degrees': cell_degrees,
        'net_degrees': net_degrees
    }

    fn = data_dir + '/' + str(sample) + f'.degree.pkl'
    print(fn)
    f = open(fn, "wb")
    pickle.dump(d, f)
    f.close()