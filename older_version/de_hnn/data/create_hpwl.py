import numpy as np
import gzip
import json
import pickle
from collections import defaultdict
from tqdm import tqdm
from multiprocessing import Pool

def log10_except_0(a):
    output = []
    for num in a:
        if num == 0:
            output.append(0)
        else:
            output.append(np.log10(num))
    output = np.array(output)
    return output

data_dir = "2023-03-06_data"
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
sample_names = []
corresponding_design = []
corresponding_variant = []
for idx in range(num_designs):
    for variant in range(num_variants_list[idx]):
        sample_name = raw_data_dir + designs_list[idx] + '/' + str(variant + 1) + '/'
        sample_names.append(sample_name)
        corresponding_design.append(designs_list[idx])
        corresponding_variant.append(variant + 1)

for sample in range(50, 74):
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
    X = [instances[idx]['xloc'] for idx in range(num_instances)]
    Y = [instances[idx]['yloc'] for idx in range(num_instances)]
    
    assert len(X) == len(Y)
    
    fn = data_dir + '/' + str(sample) + '.bipartite.pkl'
    f = open(fn, "rb")
    dictionary = pickle.load(f)
    f.close()
    
    instances = dictionary['instance_idx']
    nets = dictionary['net_idx']
    
    net_dict = defaultdict(list)
    net_X_dict = defaultdict(list)
    net_Y_dict = defaultdict(list)

    for idx in tqdm(range(len(instances))):
        net = nets[idx]
        instance = instances[idx]

        net_dict[net].append(instance)
        net_X_dict[net].append(X[instance])
        net_Y_dict[net].append(Y[instance])
        
    def hpwl(x_y):
        x_lst = x_y[0]
        y_lst = x_y[1]
        max_x = max(x_lst)
        min_x = min(x_lst)
        max_y = max(y_lst)
        min_y = min(y_lst)

        output = (max_x - min_x) + (max_y - min_y)

        return output
    
    x_y_lst = []
    
    all_nets = np.unique(nets)
    
    for net in tqdm(all_nets):
        x_lst = net_X_dict[net]
        y_lst = net_Y_dict[net]
        x_y_lst.append([x_lst, y_lst])
    
    if __name__ == '__main__':
        with Pool(10) as p:
            hpwl_net = list(tqdm(p.imap(hpwl, x_y_lst, chunksize=1000), total=len(x_y_lst)))

    hpwl = [0.0 for idx in range(num_nets)]
    
    for idx in range(len(hpwl_net)):
        net = all_nets[idx]
        hpwl[net] = hpwl_net[idx]
    
    new_hpwl = log10_except_0(hpwl)
    
    dictionary = {"hpwl":new_hpwl, 
                  "orig_hpwl":hpwl,
                  "num_nets":num_nets}
    
    fn = data_dir + '/' + str(sample) + '.net_hpwl.pkl'
    f = open(fn, "wb")
    pickle.dump(dictionary, f)
    f.close()