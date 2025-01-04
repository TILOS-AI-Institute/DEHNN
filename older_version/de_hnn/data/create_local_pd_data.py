import numpy as np
import os
import time
import scipy
import pickle
import networkx as nx

import sys

from pd_util import *

from torch_geometric.loader import DataLoader

from tqdm import tqdm

from persim import PersImage
from persim import PersistenceImager

import matplotlib.pyplot as plt

import gudhi as gd

abs_dir = '/data/zluo/'
raw_data_dir = abs_dir + 'new_data/'
graph_rep = 'star'

data_dir = '2023-03-06_data' 

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
data_dir = '2023-03-06_data'


sample = sys.argv[1] 
# Connection data 
graph_rep = 'star'
fn = data_dir + '/' + str(sample) + f'.{graph_rep}.pkl'
f = open(fn, "rb")
dictionary = pickle.load(f)
f.close()
print('Read file', fn)

edge_index = dictionary['edge_index']


def l1_dis(node1, node2, edge_attr):
    first_pos = pos_lst[node1]
    second_pos = pos_lst[node2]
    distance = np.linalg.norm((first_pos - second_pos), ord=1)
    return distance

def ego_graph(G, n, radius=1, center=True, undirected=False, distance=None):
    
    
    if undirected:
        if distance is not None:
            sp, _ = nx.single_source_dijkstra(
                G.to_undirected(), n, cutoff=radius, weight=distance
            )
        else:
            sp = dict(
                nx.single_source_shortest_path_length(
                    G.to_undirected(), n, cutoff=radius
                )
            )
    else:
        if distance is not None:
            sp, _ = nx.single_source_dijkstra(G, n, cutoff=radius)
        else:
            sp = dict(nx.single_source_shortest_path_length(G, n, cutoff=radius))
    
    H = list(G.subgraph(sp).edges)
    return H, sp



def gen_local_pd(G, s_n, radius, processed=True, distance=None):
    edgs, loc_delay_dict = ego_graph(G, s_n, radius, distance=distance)
    neigh_lst = [0.0 for idx in range(radius)]
    
    if len(edgs) < 2 or len(loc_delay_dict) < 2:
        return neigh_lst, np.array([[0., 0.]]), np.array([[0., 0.]])
    
    st = gd.SimplexTree()

    for edg in edgs:
        st.insert(edg)
    
    
    for node, delay in loc_delay_dict.items():
        
        dis = l1_dis(s_n, node, None)
        st.assign_filtration([node], dis)
        
        if delay == 0:
            continue
        else:
            neigh_lst[delay - 1] += 1.0 
            
    st.extend_filtration()
    dgms = st.extended_persistence()
    
    if processed:
        zero_dgm = np.array([list([tp[1][0], tp[1][1]]) for tp in dgms[0]] + [list([tp[1][0], tp[1][1]]) for tp in dgms[2]])
        one_dgm = np.array([list([tp[1][1], tp[1][0]]) for tp in dgms[-1]])
        
        if zero_dgm.shape[-1] != 2:
            zero_dgm = np.array([[0., 0.]])
                            
        if one_dgm.shape[-1] != 2:
            one_dgm = np.array([[0., 0.]])
        
    return neigh_lst, zero_dgm, one_dgm

# Connection data 
graph_rep = 'node_features'
fn = data_dir + '/' + str(sample) + f'.{graph_rep}.pkl'
f = open(fn, "rb")
d = pickle.load(f)
f.close()
print('Read file', fn)

instance_features = d['instance_features']

G_in = nx.DiGraph(edge_index.tolist())
G_out = G_in.reverse()

nodelist = list(G_in.nodes)

num_nodes = d['num_instances']

# if instance_features.shape[1] >= 10:
#     print('Already generated degree')
# else:
#     print('Start generating degree')
#     in_degree = [0 for i in range(num_nodes)]
#     out_degree = [0 for i in range(num_nodes)]

#     for node in tqdm(nodelist):
#         in_degree[node] = G_in.in_degree[node]
#         out_degree[node] = G_in.out_degree[node]

#     instance_features = np.concatenate([instance_features, np.array(in_degree).reshape(-1, 1), np.array(out_degree).reshape(-1, 1)], axis=1)

#     d['instance_features'] = instance_features

global pos_lst
pos_lst = instance_features[:, :2]

pd_dictionary = {}

from collections import defaultdict

for idx in range(2):
    if idx == 1:
        G = G_in
    else:
        G = G_out
    
    radius = 6
    nodelist = list(G.nodes)
    num_nodes = len(pos_lst)
    #all_zero_dgms = [[0.0] for i in range(num_nodes)]
    all_one_dgms = [[0.0 for idx in range(25)] for i in range(num_nodes)]
    all_neigh = [[0.0 for idx in range(radius)] for i in range(num_nodes)]

    #pimgr_0 = PersistenceImager(pixel_size=3, birth_range=(0, 6), pers_range=(0, 6))
    pimgr_1 = PersistenceImager(pixel_size=20, birth_range=(0, 100), pers_range=(0, 100))

    for node in tqdm(nodelist):
        if G.degree[node] >= 100:
            neigh_lst, zero_dgm, one_dgm  = gen_local_pd(G, node, radius)
            zero_dgm = zero_dgm/0.01
            one_dgm = one_dgm/0.01
            #L_0 = pimgr_0.transform(zero_dgm)
            L_1 = pimgr_1.transform(one_dgm)
            #all_zero_dgms[node] = L_0.flatten().tolist()
            all_one_dgms[node] = L_1.flatten().tolist()
            all_neigh[node] = neigh_lst
            G.remove_node(node)

    nodelist = list(G.nodes)
    for node in tqdm(nodelist):
        neigh_lst, zero_dgm, one_dgm = gen_local_pd(G, node, radius)
        if neigh_lst[0] == 0:
            continue
        zero_dgm = zero_dgm/0.01
        one_dgm = one_dgm/0.01
        #L_0 = pimgr_0.transform(zero_dgm)
        L_1 = pimgr_1.transform(one_dgm)
        #all_zero_dgms[node] = L_0.flatten().tolist()
        all_one_dgms[node] = L_1.flatten().tolist()
        all_neigh[node] = neigh_lst

    #pd_dictionary[f"all_zero_dgms_{idx}"] = np.array(all_zero_dgms)
    pd_dictionary[f"all_one_dgms_{idx}"] = np.array(all_one_dgms)
    pd_dictionary[f"all_neigh_{idx}"] = np.array(all_neigh)
    
pd_d = pd_dictionary

pd = np.concatenate([pd_d['all_one_dgms_0'], pd_d['all_one_dgms_1']], axis=1)
    
neighbor = np.concatenate([pd_d['all_neigh_0'], pd_d['all_neigh_1']], axis=1)

dictionary = {
    "pd": pd,
    "neighbor": neighbor
}

print(sample, pd.shape, neighbor.shape)
with open(f"2023-03-06_data/{sample}.node_neighbor_features.pkl", "wb") as f:
    pickle.dump(dictionary, f)
