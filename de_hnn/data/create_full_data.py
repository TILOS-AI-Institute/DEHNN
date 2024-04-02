import numpy as np
import gzip
import json
from scipy.stats import binned_statistic_2d
import time
import pickle

import networkx as nx

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

# all_idx_lst = [] #key
# first_idx_lst = [] #value

# first_idx = 0
# all_idx = 0

# for num in num_variants_list:
#     for i in range(num):
#         first_idx_lst.append(first_idx)
#         all_idx_lst.append(all_idx)
#         all_idx += 1
        
#     first_idx += num
    
# idx_dict = {}

# for i in range(len(all_idx_lst)):
#     idx_dict[all_idx_lst[i]] = first_idx_lst[i]
    
# with open(f"2023-03-06_data/all.idx_to_design.pkl", "wb") as f:
#     pickle.dump(idx_dict, f)

# assert num_designs == len(num_variants_list)

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
print(N)
data_dir = '2023-03-06_data/'

# +--------------------+
# | Global information |
# +--------------------+

# Read the csv file
with open(raw_data_dir + '/settings.csv') as f:
    lines = f.readlines()

lines = lines[1:]
for line in lines:
    words = line.strip().split(',')
    dictionary = {
        'design': words[0],
        'variant': int(words[1]),
        'core_utilization': float(words[2]),
        'max_routing_layer': words[3],
        'clk_per': int(words[4]),
        'clk_uncertainty': float(words[5]),
        'flow_stage': words[6],
        'hstrap_layer': words[7],
        'hstrap_width': float(words[8]),
        'hstrap_pitch': float(words[9]),
        'vstrap_layer': words[10],
        'vstrap_width': float(words[11]),
        'vstrap_pitch': float(words[12])
    }

    sample_idx = -1
    for idx in range(N):
        if dictionary['design'] == corresponding_design[idx] and dictionary['variant'] == corresponding_variant[idx]:
            sample_idx = idx
            break

    fn = data_dir + str(sample_idx) + '.global_information.pkl'
    f = open(fn, "wb")
    pickle.dump(dictionary, f)
    f.close()
    print('Save file', fn)

# +-------------------------+
# | Information about cells |
# +-------------------------+

cells_fn = raw_data_dir + 'cells.json.gz'
with gzip.open(cells_fn, 'r') as fin:
    cell_data = json.load(fin)

widths = []
heights = []
for idx in range(len(cell_data)):
    width = cell_data[idx]['width']
    height = cell_data[idx]['height']
    widths.append(width)
    heights.append(height)

widths = np.array(widths)
heights = np.array(heights)

min_cell_width = np.min(widths)
max_cell_width = np.max(widths)
min_cell_height = np.min(heights)
max_cell_height = np.max(heights)

print('min cell width:', min_cell_width)
print('max cell width:', max_cell_width)
print('mean cell width:', np.mean(widths))
print('std cell width:', np.std(widths))
print()
print('min cell height:', min_cell_height)
print('max cell height:', max_cell_height)
print('mean cell height:', np.mean(heights))
print('std cell height:', np.std(heights))

widths = (widths - min_cell_width) / (max_cell_width - min_cell_width)
heights = (heights - min_cell_height) / (max_cell_height - min_cell_height)
print('Done processing cell sizes')

cell_to_edge_dict = {item['id']:{inner_item['id']: inner_item['dir'] for inner_item in item['terms']} for item in cell_data}
print('Done processing edge types dict')

# For each sample
for sample in range(50, 74):
    
    # +--------------------------------------+
    # | Information about instances and nets |
    # +--------------------------------------+

    folder = sample_names[sample]
    design = corresponding_design[sample]
    instances_nets_fn = folder + design + '.json.gz'

    print('--------------------------------------------------')
    print('Folder:', folder)
    print('Design:', design)
    print('Instances & nets info:', instances_nets_fn)

    with gzip.open(instances_nets_fn, 'r') as fin:
        instances_nets_data = json.load(fin)

    instances = instances_nets_data['instances']
    nets = instances_nets_data['nets']

    inst_to_cell = {item['id']:item['cell'] for item in instances}

    num_instances = len(instances)
    num_nets = len(nets)

    print('Number of instances:', num_instances)
    print('Number of nets:', num_nets)
    
    # Get placements info
    xloc_list = [instances[idx]['xloc'] for idx in range(num_instances)]
    yloc_list = [instances[idx]['yloc'] for idx in range(num_instances)]
    cell = [instances[idx]['cell'] for idx in range(num_instances)]
    cell_width = [widths[cell[idx]] for idx in range(num_instances)]
    cell_height = [heights[cell[idx]] for idx in range(num_instances)]
    orient = [instances[idx]['orient'] for idx in range(num_instances)]
    
    x_min = min(xloc_list)
    x_max = max(xloc_list)
    y_min = min(yloc_list)
    y_max = max(yloc_list)

    print('min xloc:', x_min)
    print('max xloc:', x_max)
    print('min yloc:', y_min)
    print('max yloc:', y_max)

    X = np.expand_dims(np.array(xloc_list), axis = 1)
    Y = np.expand_dims(np.array(yloc_list), axis = 1)
    X = (X - x_min) / (x_max - x_min)
    Y = (Y - y_min) / (y_max - y_min)

    cell = np.expand_dims(np.array(cell), axis = 1)
    cell_width = np.expand_dims(np.array(cell_width), axis = 1)
    cell_height = np.expand_dims(np.array(cell_height), axis = 1)
    orient = np.expand_dims(np.array(orient), axis = 1)

    instance_features = np.concatenate((X, Y, cell, cell_width, cell_height, orient), axis = 1)
    
    dictionary = {
        'num_instances': num_instances,
        'num_nets': num_nets,
        'x_min': x_min,
        'x_max': x_max,
        'y_min': y_min,
        'y_max': y_max,
        'min_cell_width': min_cell_width,
        'max_cell_width': max_cell_width,
        'min_cell_height': min_cell_height,
        'max_cell_height': max_cell_height,
        'instance_features': instance_features,
        'sample_name': sample_name,
        'folder': folder,
        'design': design
    }
    fn = data_dir + '/' + str(sample) + '.node_features.pkl'
    f = open(fn, "wb")
    pickle.dump(dictionary, f)
    f.close()
    print('Save file', fn)

    # +-------------------------+
    # | Get the connection data |
    # +-------------------------+

    connection_fn = folder + design + '_connectivity.npz'
    connection_data = np.load(connection_fn)
    print('Connection info:', connection_fn)
    
    # get the direction of each edge between inst and net
    dirs = []
    edge_t = connection_data['data']
    instance_idx = connection_data['row']
    
    for idx in range(len(instance_idx)):
        inst = instance_idx[idx]
        cell = inst_to_cell[inst]
        edge_dict = cell_to_edge_dict[cell]
        t = edge_t[idx]
        direction = edge_dict[t]
        dirs.append(direction)

    dirs = np.array(dirs)
    
    assert dirs.shape == connection_data['data'].shape
    
    dictionary = {
        'instance_idx': connection_data['row'],
        'net_idx': connection_data['col'],
        'edge_attr': connection_data['data'],
        'edge_dir': dirs, 
        'sample_name': sample_name,
        'folder': folder,
        'design': design
    }
    
    edge_index = np.array([dictionary['instance_idx'], dictionary['net_idx']]).T
    edge_dir = dictionary['edge_dir']
    n_edge_index = []
    for idx in range(len(edge_index)):
        tp = edge_index[idx]
        direct = edge_dir[idx]

        if direct == 0:
            n_edge_index.append([tp[1], tp[0]])
        else:
            n_edge_index.append([tp[0], tp[1]])

    n_edge_index = np.array(n_edge_index).T
   
    dictionary['edge_index'] = n_edge_index

    
    fn = data_dir + '/' + str(sample) + '.bipartite.pkl'
    f = open(fn, "wb")
    pickle.dump(dictionary, f)
    f.close()
    print('Save file', fn)
    
    
    instances = dictionary['instance_idx']
    nets = dictionary['net_idx']
    directs = dictionary['edge_dir']
    attrs = dictionary['edge_attr']
    
    drive_dict = dict()
    for idx in range(len(instances)):
        direct = directs[idx]
        net = nets[idx]

        if direct == 1:
            assert net not in drive_dict

            drive_dict[net] = idx
        else:
            continue



    print("finished building drive and net dict")


    n_row = []
    n_col = []
    n_edge_attr = []
    b_nets = []
    b_terms = []
    for idx in range(len(nets)):
        net = nets[idx]
        inst = instances[idx]
        attr = attrs[idx]
        
        if net not in drive_dict:
            b_nets.append(net)
            b_terms.append(idx)
        else:
            inst_idx = drive_dict[net]
            drive_inst = instances[inst_idx]
            
            if inst == drive_inst:
                continue
                
            n_row.append(drive_inst)
            n_col.append(inst)
            n_edge_attr.append(attr)
    
    
    #dictionary['edge_attr'] = np.array(n_edge_attr)
    edge_index = np.array([n_row, n_col]).T
    dictionary['edge_index'] = edge_index
    dictionary['b_nets'] = b_nets
    dictionary['b_terms'] = b_terms
    
    
    
    pos_lst = instance_features[:, :2] 
    for idx in range(len(edge_index)):
        tp = edge_index[idx]
        direct = edge_dir[idx]

        # compute the edge distances based on nodes' positions
        first_pos = pos_lst[tp[0]]
        second_pos = pos_lst[tp[1]]
        l1_dis = np.linalg.norm((first_pos - second_pos), ord=1)
        new_attr = [l1_dis]
        if idx == 0:
            print(new_attr)
        n_edge_attr[idx] = new_attr
    

    n_edge_attr = np.array(n_edge_attr)
    print(n_edge_attr.shape, n_edge_attr[-1])
    dictionary['edge_attr'] = n_edge_attr
    

    fn = data_dir + '/' + str(sample) + '.star.pkl'
    f = open(fn, "wb")
    pickle.dump(dictionary, f)
    f.close()
    print('Save file', fn)


    # +---------------------+
    # | Get congestion data |
    # +---------------------+

    congestion_fn = folder + design + '_congestion.npz'
    congestion_data = np.load(congestion_fn)
    print('Congestion info:', congestion_fn)

    congestion_data_demand = congestion_data['demand']
    congestion_data_capacity = congestion_data['capacity']

    num_layers = len(list(congestion_data['layerList']))
    print('Number of layers:', num_layers)
    print('Layers:', list(congestion_data['layerList']))

    ybl = congestion_data['yBoundaryList']
    xbl = congestion_data['xBoundaryList']

    all_demand = []
    all_capacity = []

    for layer in list(congestion_data['layerList']):
        print('Layer', layer, ':')
        lyr = list(congestion_data['layerList']).index(layer)

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

        print('    Number of demand > capacity:', count_congestions)
        print('    Average capacity - demand:', average_diff)
        print('    Average demand:', average_demand)
        print('    Average capacity:', average_capacity)

    demand = np.concatenate(all_demand, axis = 1)
    capacity = np.concatenate(all_capacity, axis = 1)

    dictionary = {
        'demand': demand,
        'capacity': capacity
    }
    fn = data_dir + '/' + str(sample) + '.targets.pkl'
    f = open(fn, "wb")
    pickle.dump(dictionary, f)
    f.close()
    print('Save file', fn)

print('Done')
