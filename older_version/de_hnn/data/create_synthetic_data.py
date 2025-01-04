import numpy as np
import gzip
import json
from scipy.stats import binned_statistic_2d
import time
import pickle

# Synthetic data
N = 5
data_dir = 'synthetic_data/'

# Folder
# folder = 'NCSU-DigIC-GraphData-2022-10-15/'
folder = 'RosettaStone-GraphData-2023-02-27/'

# Design
# design = 'counter'
# design = 'xbar'
# design = 'RocketTile'

design = 'superblue18'

print('Folder:', folder)
print('Design:', design)

# Information about instances and nets
instances_nets_fn = folder + '/' + design + '/1/' + design + '.json.gz'

with gzip.open(instances_nets_fn, 'r') as fin:
    instances_nets_data = json.load(fin)

instances = instances_nets_data['instances']
nets = instances_nets_data['nets']

num_instances = len(instances)
num_nets = len(nets)

print('Number of instances:', num_instances)
print('Number of nets:', num_nets)

print(instances[0])

# Get the connection data
connection_fn = folder + '/' + design + '/1/' + design + '_connectivity.npz'

connection_data = np.load(connection_fn)

# Get placements info
t = time.time()
xloc_list = [instances[idx]['xloc'] for idx in range(num_instances)]
yloc_list = [instances[idx]['yloc'] for idx in range(num_instances)]
print('Time for getting placement info:', time.time() - t)

# Divide the floor into N x N parts
x_min = min(xloc_list)
x_max = max(xloc_list)
y_min = min(yloc_list)
y_max = max(yloc_list)

print('min xloc:', x_min)
print('max xloc:', x_max)
print('min yloc:', y_min)
print('max yloc:', y_max)

x_window = (x_max - x_min) // N
y_window = (y_max - y_min) // N

x_bins = []
x = x_min
for i in range(N):
    if i > 0:
        x_bins.append(x)
    x += x_window
x_bins.append(x_max + 1)
print('x bins:', x_bins)

y_bins = []
y = y_min
for i in range(N):
    if i > 0:
        y_bins.append(y)
    y += y_window
y_bins.append(y_max + 1)
print('y bins:', y_bins)

num_samples = N * N
sample_index = []
for idx in range(num_instances):
    x = xloc_list[idx]
    y = yloc_list[idx]
    
    row_index = -1
    for i in range(N):
        if x < x_bins[i]:
            row_index = i
            break
    assert row_index >= 0
    assert row_index < N

    col_index = -1
    for i in range(N):
        if y < y_bins[i]:
            col_index = i
            break
    assert col_index >= 0
    assert col_index < N

    index = row_index * N + col_index
    sample_index.append(index)
sample_index = np.array(sample_index)

# Create connection for the synthetic dataset
instance_idx = [[] for sample in range(num_samples)]
net_idx = [[] for sample in range(num_samples)]
edge_attr = [[] for sample in range(num_samples)]

num_edges = connection_data['row'].shape[0]
print('Number of hyper-edges:', num_edges)

row = connection_data['row']
col = connection_data['col']
edge = connection_data['data']

mark = np.zeros([num_edges])
for e in range(num_edges):
    mark[e] = sample_index[row[e]]

for sample in range(num_samples):
    instance_idx[sample] = row[mark == sample]
    net_idx[sample] = col[mark == sample]
    edge_attr[sample] = edge[mark == sample]

assert num_edges == np.sum(np.array([len(instance_idx[sample]) for sample in range(num_samples)]))
assert num_edges == np.sum(np.array([len(net_idx[sample]) for sample in range(num_samples)]))
assert num_edges == np.sum(np.array([len(edge_attr[sample]) for sample in range(num_samples)]))

def standardize_array(arr):
    unique = np.unique(arr).tolist()
    return np.array([unique.index(arr[i]) for i in range(arr.shape[0])])

for sample in range(num_samples):
    dictionary = {
        'instance_idx': instance_idx[sample], # standardize_array(instance_idx[sample]),
        'net_idx': net_idx[sample],
        'edge_attr': edge_attr[sample]
    }
    fn = data_dir + '/' + str(sample) + '.bipartite.pkl'
    f = open(fn, "wb")
    pickle.dump(dictionary, f)
    f.close()
print('Done creating the bipartite representation for the synthetic dataset')

'''
for sample in range(num_samples):
    v1 = []
    v2 = []
    E = net_idx[sample].shape[0]
    start = 0
    while start < E:
        finish = start
        for i in range(start, E):
            if net_idx[sample][i] == net_idx[sample][start]:
                finish = i
            else:
                break
        for x in range(start, finish + 1):
            for y in range(x + 1, finish + 1):
                a = instance_idx[sample][x]
                b = instance_idx[sample][y]
                v1.append(a)
                v2.append(b)
                v1.append(b)
                v2.append(a)
        start = finish + 1
    dictionary = {
        'v1': v1,
        'v2': v2
    }
    fn = data_dir + '/' + str(sample) + '.clique.pkl'
    f = open(fn, "wb")
    pickle.dump(dictionary, f)
    f.close()
print('Done creating the clique representation for the synthetic dataset')
'''

# Create node features for the synthetic dataset
x1 = min(xloc_list)
y1 = min(yloc_list)
x2 = max(xloc_list)
y2 = max(yloc_list)

check_sum = 0
for sample in range(num_samples):
    binary = (sample_index == sample)
    count = np.sum(binary)
    check_sum += count
    
    print('------------------------')
    print('Sample', sample, ':', count, 'instances')
    
    # Node features
    indices = np.array([idx for idx in range(num_instances) if sample_index[idx] == sample])
    
    X = []
    Y = []
    cell = []
    orient = []

    for idx in indices:
        X.append(xloc_list[idx])
        Y.append(yloc_list[idx])
        cell.append(instances[idx]['cell'])
        orient.append(instances[idx]['orient'])

    assert len(X) == count
    assert len(Y) == count

    minX = min(X)
    maxX = max(X)
    minY = min(Y)
    maxY = max(Y)

    X = np.expand_dims(np.array(X), axis = 1)
    Y = np.expand_dims(np.array(Y), axis = 1)
    X = (X - x1) / (x2 - x1)
    Y = (Y - y1) / (y2 - y1)
    
    cell = np.expand_dims(np.array(cell), axis = 1)
    orient = np.expand_dims(np.array(orient), axis = 1)

    instance_features = np.concatenate((X, Y, cell, orient), axis = 1)
    
    dictionary = {
        'num_instances': count,
        'num_nets': num_nets,
        'minX': minX,
        'maxX': maxX,
        'minY': minY,
        'maxY': maxY,
        'instance_features': instance_features
    }
    fn = data_dir + '/' + str(sample) + '.node_features.pkl'
    f = open(fn, "wb")
    pickle.dump(dictionary, f)
    f.close()
    
assert check_sum == num_instances
print('Done creating the node features')

# Congestion data
congestion_fn = folder + '/' + design + '/1/' + design + '_congestion.npz'

# Load congestion data file
congestion_data = np.load(congestion_fn)

congestion_data_demand = congestion_data['demand']
congestion_data_capacity = congestion_data['capacity']

num_layers = len(list(congestion_data['layerList']))
print('Number of layers:', num_layers)
print('Layers:', list(congestion_data['layerList']))

all_demand = []
all_capacity = []

for layer in list(congestion_data['layerList']):
    print(layer, '--------------------------')
    lyr = list(congestion_data['layerList']).index(layer)
    ybl = congestion_data['yBoundaryList']
    xbl = congestion_data['xBoundaryList']

    # Binned statistics 2D
    t = time.time()
    ret = binned_statistic_2d(xloc_list, yloc_list, None, 'count', bins = [xbl[1:], ybl[1:]], expand_binnumbers = True)
    print('Time for binned statistics:', time.time() - t)

    i_list = np.array([ret.binnumber[0, idx] - 1 for idx in range(num_instances)])
    j_list = np.array([ret.binnumber[1, idx] - 1 for idx in range(num_instances)])

    # Get demand and capacity
    t = time.time()
    demand_list = congestion_data_demand[lyr, i_list, j_list].flatten()
    capacity_list = congestion_data_capacity[lyr, i_list, j_list].flatten()
    print('Time to get demand and capacity:', time.time() - t)

    demand_list = np.array(demand_list)
    capacity_list = np.array(capacity_list)

    all_demand.append(np.expand_dims(demand_list, axis = 1))
    all_capacity.append(np.expand_dims(capacity_list, axis = 1))

    average_demand = np.mean(demand_list)
    average_capacity = np.mean(capacity_list)
    average_diff = np.mean(capacity_list - demand_list)
    count_congestions = np.sum(demand_list > capacity_list)

    print()
    print('Number of demand > capacity:', count_congestions)
    print('Average capacity - demand:', average_diff)
    print('Average demand:', average_demand)
    print('Average capacity:', average_capacity)

demand = np.concatenate(all_demand, axis = 1)
capacity = np.concatenate(all_capacity, axis = 1)

for sample in range(num_samples):
    indices = np.array([idx for idx in range(num_instances) if sample_index[idx] == sample])
    dictionary = {
        'demand': demand[indices, :],
        'capacity': capacity[indices, :]
    }
    fn = data_dir + '/' + str(sample) + '.targets.pkl'
    f = open(fn, "wb")
    pickle.dump(dictionary, f)
    f.close()

print('Done')
