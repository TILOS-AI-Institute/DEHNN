import numpy as np
import pickle

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
    'superblue19'
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

print(sample_names)
print(corresponding_design)
print(corresponding_variant)

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

print('Done')
