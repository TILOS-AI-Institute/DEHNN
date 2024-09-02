import torch
import torch.nn
from torch_geometric.data import Dataset
from torch_geometric.data import Data

import numpy as np
import pickle

from tqdm import tqdm

class pyg_dataset(Dataset):
    def __init__(self, data_dir, fold_index, split, target, load_pe = False, num_eigen = 10, load_global_info = False, load_pd = False, vn = False, concat=False, net=True, design = 19, pl = 0):
        super().__init__()
        self.data_dir = data_dir
        self.fold_index = fold_index
        self.split = split
        self.target = target
        assert target == 'demand' or target == 'capacity' or target == 'congestion' or target == 'hpwl'
        print('Learning target:', self.target)

        # Position encoding
        self.load_pe = load_pe
        self.num_eigen = num_eigen
        self.load_global_info = load_global_info
        self.load_pd = load_pd

        # Read cross-validation
        print(f"Placement Information {pl}")
        file_name = data_dir + f'cross_validation/9_fold_cross_validation_{design}.pkl'
        f = open(file_name, 'rb')
        dictionary = pickle.load(f)
        f.close()
        fix_folds = dictionary['fix_folds']

        train_indices = fix_folds[0]
        valid_indices = fix_folds[1]
        test_indices = fix_folds[2]

        if self.split == 'train':
            self.sample_indices = train_indices
        elif self.split == 'valid':
            self.sample_indices = valid_indices
        else:
            self.sample_indices = test_indices
        
        self.num_samples = len(self.sample_indices)
        print('Number of samples:', self.num_samples)

        # Re data
        self.data = []

        for sample in tqdm(self.sample_indices):
            print(sample)
            graph_index = sample
            # Read node features
            file_name = data_dir + '/' + str(sample) + '.node_features.pkl'
            f = open(file_name, 'rb')
            dictionary = pickle.load(f)
            f.close()

            self.design_name = dictionary['design']
            num_instances = dictionary['num_instances']
            num_nets = dictionary['num_nets']
            instance_features = torch.Tensor(dictionary['instance_features'])
            
            if pl:
                instance_features = instance_features
            else:
                #pos_lst = dictionary['instance_features'][:, :2]
                #X, Y = pos_lst.T[0], pos_lst.T[1]
                #xloc_list = X*(dictionary['x_max'] - dictionary['x_min']) + dictionary['x_min']
                #yloc_list = Y*(dictionary['y_max'] - dictionary['y_min']) + dictionary['y_min']
                #instance_features = torch.cat([instance_features[:, 2:], torch.Tensor(xloc_list).unsqueeze(dim = 1), torch.Tensor(yloc_list).unsqueeze(dim = 1)], dim=1) 
                instance_features = instance_features[:, 2:]
            
            # Read learning targets
            file_name = data_dir + '/' + str(sample) + '.net_demand_capacity.pkl'
            f = open(file_name, 'rb')
            dictionary = pickle.load(f)
            f.close()
            
            if target == 'hpwl':
                fn = data_dir + '/' + str(graph_index) + '.net_hpwl.pkl'
                f = open(fn, "rb")
                d_hpwl = pickle.load(f)
                f.close()
                hpwl = torch.Tensor(d_hpwl['hpwl'])

            demand = torch.Tensor(dictionary['demand'])
            capacity = torch.Tensor(dictionary['capacity'])

            if self.target == 'demand':
                y = demand.unsqueeze(dim = 1)
            elif self.target == 'capacity':
                y = capacity.unsqueeze(dim = 1)
            elif self.target == 'congestion':
                congestion = demand - capacity
                y = congestion.unsqueeze(dim = 1)
            elif self.target == 'hpwl':
                y = hpwl.unsqueeze(dim=1)
            else:
                print('Unknown learning target')
                assert False

            # Read connection
            file_name = data_dir + '/' + str(sample) + '.bipartite.pkl'
            f = open(file_name, 'rb')
            dictionary = pickle.load(f)
            f.close()
            
            instance_idx = torch.Tensor(dictionary['instance_idx']).unsqueeze(dim = 1).long()
            net_idx = torch.Tensor(dictionary['net_idx']) + num_instances
            net_idx = net_idx.unsqueeze(dim = 1).long()
            edge_attr = torch.Tensor(dictionary['edge_attr']).float().unsqueeze(dim = 1).float()

            edge_index = torch.cat((instance_idx, net_idx), dim = 1)

            edge_index = torch.transpose(edge_index, 0, 1)
            
            x = instance_features

            # PyG data
            example = Data()
            example.__num_nodes__ = x.size(0)
            example.x = x

#             # Load capacity
            capacity = capacity.unsqueeze(dim = 1)
            norm_cap = (capacity - torch.min(capacity)) / (torch.max(capacity) - torch.min(capacity))
            capacity_features = torch.cat([capacity, torch.sqrt(capacity), norm_cap, torch.sqrt(norm_cap), torch.square(norm_cap), torch.sin(norm_cap), torch.cos(norm_cap)], dim = 1)

            example.x_net = capacity_features
            example.num_instances = num_instances

            example.y = y
            example.edge_index_node_net = edge_index
            example.edge_index_net_node = edge_index.flip([0])

            with open(f"../../data/2023-03-06_data/all.idx_to_design.pkl", "rb") as f:
                idx_dict = pickle.load(f)

            first_index = idx_dict[graph_index]

            with open(f"../../data/2023-03-06_data/{first_index}.nn_conn.pkl", "rb") as f:
                conn_dict = pickle.load(f)

            node_node_conn = conn_dict["nn_edge_index"]
            example.edge_index_node_node = node_node_conn
            example.edge_attr = edge_attr[:2]

            fn = data_dir + '/' + str(graph_index) + f'.degree.pkl'
            f = open(fn, "rb")
            d = pickle.load(f)
            f.close()
            
            example.cell_degrees = torch.Tensor(d['cell_degrees'])
            example.net_degrees = torch.Tensor(d['net_degrees']) 
            example.x = torch.cat([example.x, example.cell_degrees.unsqueeze(dim = 1)], dim = 1)
            
            if not pl:
                example.x_net = example.net_degrees.unsqueeze(dim = 1)
            else:
                example.x_net = torch.cat([example.x_net, example.net_degrees.unsqueeze(dim = 1)], dim = 1)

            # Load positional encoding
            if self.load_pe == True:
                file_name = data_dir + '/' + str(first_index) + '.eigen.' + str(self.num_eigen) + '.pkl'
                f = open(file_name, 'rb')
                dictionary = pickle.load(f)
                f.close()

                example.evects = torch.Tensor(dictionary['evects'])
                evals = torch.Tensor(dictionary['evals'])

            # Load global information
            if self.load_global_info == True:
                num_nodes = example.x.size(0)
                global_info = torch.Tensor(np.concatenate([[num_nodes], evals]))
                global_info = torch.cat([global_info.unsqueeze(dim = 0) for i in range(num_nodes)], dim = 0)
                example.x = torch.cat([example.x, global_info], dim = 1)

            if pl:
                nerighbor_f = 'node_neighbors/'
            else:
                nerighbor_f = 'node_neighbors/'
                
            # Load persistence diagram and neighbor list
            if self.load_pd == True:
                file_name = data_dir + '/' + nerighbor_f + str(first_index) + '.node_neighbor_features.pkl'
                f = open(file_name, 'rb')
                dictionary = pickle.load(f)
                f.close()

                pd = torch.Tensor(dictionary['pd'])
                neighbor_list = torch.Tensor(dictionary['neighbor'])

                assert pd.size(0) == num_instances
                assert neighbor_list.size(0) == num_instances

                example.x = torch.cat([example.x, pd, neighbor_list], dim = 1)
            else:
                file_name = data_dir + '/' + nerighbor_f + str(first_index) + '.node_neighbor_features.pkl'
                f = open(file_name, 'rb')
                dictionary = pickle.load(f)
                f.close()

                neighbor_list = torch.Tensor(dictionary['neighbor'])

                num_neighbor = int(neighbor_list.shape[1]/2)

                neighbor_list = neighbor_list[:, :num_neighbor] + neighbor_list[:, num_neighbor:]
                
                assert neighbor_list.size(0) == num_instances
                assert neighbor_list.size(1) == 6
                example.x = torch.cat([example.x, neighbor_list], dim = 1)  

            if concat:
                f = open(data_dir + '/' + str(graph_index) + '.net_features.pkl', 'rb')
                dictionary = pickle.load(f)
                f.close()
                node_feat = dictionary['instance_features']

                example.evects = example.evects[num_instances:]
                example.evals = example.evals[num_instances:]

                example.x = torch.Tensor(node_feat)
                example.x_net = None
                

            self.data.append(example)

        print('Done reading data')

    def len(self):
        return self.num_samples

    def get(self, idx):
        return self.data[idx]

