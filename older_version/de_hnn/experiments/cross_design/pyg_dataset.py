import torch
import torch.nn
from torch_geometric.data import Dataset
from torch_geometric.data import Data

import numpy as np
import pickle

from tqdm import tqdm

class pyg_dataset(Dataset):
    def __init__(self, data_dir, fold_index, split, target, load_pe = False, num_eigen = 10, load_global_info = True, load_pd = False, design = 19, pl = 0, vn = False, concat=False, net=False):
        super().__init__()
        self.data_dir = data_dir
        self.fold_index = fold_index
        self.split = split
        self.target = target
        assert target == 'demand' or target == 'capacity' or target == 'congestion' or target == 'classify'
        print('Learning target:', self.target)

        # Position encoding
        self.load_pe = load_pe
        self.num_eigen = num_eigen
        self.load_global_info = load_global_info
        self.load_pd = load_pd

        # Read cross-validation
        pl = bool(pl)
        print(f"Placement Information {pl}")
        file_name = data_dir + f'/cross_validation/9_fold_cross_validation_{design}.pkl'
        f = open(file_name, 'rb')
        dictionary = pickle.load(f)
        f.close()
        #folds = dictionary['folds']
        fix_folds = dictionary['fix_folds']

        # Take the sample indices
#         test_indices = folds[self.fold_index]
#         train_indices = [idx for idx in range(total_samples) if idx not in test_indices]
        
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

        # Read data
        self.data = []

        for sample in tqdm(self.sample_indices):
            graph_index = sample
            print(sample)
            # Read node features
            file_name = data_dir + '/' + str(sample) + '.node_features.pkl'
            f = open(file_name, 'rb')
            dictionary = pickle.load(f)
            f.close()

            self.design_name = dictionary['design']

            num_instances = dictionary['num_instances']
            num_nets = dictionary['num_nets']
            instance_features = torch.Tensor(dictionary['instance_features'])
            
            print(instance_features.shape)
            #if not pl:
            instance_features = instance_features[:, 2:]
            
            net_features = torch.zeros(num_nets, instance_features.size(1))

            print(instance_features.shape)
            # Read learning targets
            file_name = data_dir + '/' + str(sample) + '.targets.pkl'
            f = open(file_name, 'rb')
            dictionary = pickle.load(f)
            f.close()

            demand = torch.Tensor(dictionary['demand'])
            capacity = torch.Tensor(dictionary['capacity'])

            if self.target == 'demand':
                y = demand.unsqueeze(dim = 1)
            elif self.target == 'capacity':
                y = capacity.unsqueeze(dim = 1)
            elif self.target == 'congestion':
                congestion = demand - capacity
                y = congestion.unsqueeze(dim = 1)
            elif self.target == 'classify':
                y = dictionary['classify']
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
            print(x.shape)
            example.x = x

            # Load capacity
            #if graph_index < 50:
            if pl:
                capacity = capacity.unsqueeze(dim = 1)
                norm_cap = (capacity - torch.min(capacity)) / (torch.max(capacity) - torch.min(capacity))
                capacity_features = torch.cat([capacity, torch.sqrt(capacity), norm_cap, torch.sqrt(norm_cap), torch.square(norm_cap), torch.sin(norm_cap), torch.cos(norm_cap)], dim = 1)
            #example.x = torch.cat([example.x, capacity_features], dim = 1)

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

            example.cell_degrees = torch.tensor(d['cell_degrees'])
            example.net_degrees = torch.tensor(d['net_degrees'])

            #if net:
                #example.x_net = torch.cat([capacity_features, example.net_degrees.unsqueeze(dim = 1)], dim = 1)
            #else:
            example.x_net = example.net_degrees.unsqueeze(dim = 1)

            if vn:       
                if pl:       
                    file_name = data_dir + '/' + str(first_index) + '.metis_part_dict.pkl'
                else:
                    file_name = data_dir + '/' + str(graph_index) + '.star_part_dict.pkl'
                f = open(file_name, 'rb')
                part_dict = pickle.load(f)
                f.close()
      
                part_id_lst = []
      
                for idx in range(len(example.x)):
                    part_id_lst.append(part_dict[idx])
      
                part_id = torch.LongTensor(part_id_lst)
                
                example.num_vn = len(torch.unique(part_id))
      
                top_part_id = torch.Tensor([0 for idx in range(example.num_vn)]).long()
      
                example.num_top_vn = len(torch.unique(top_part_id))
      
                example.part_id = part_id
                example.top_part_id = top_part_id
            
            # Load positional encoding
            if self.load_pe == True:
                file_name = data_dir + '/' + str(first_index) + '.eigen.' + str(self.num_eigen) + '.pkl'
                f = open(file_name, 'rb')
                dictionary = pickle.load(f)
                f.close()

                example.evects = torch.Tensor(dictionary['evects'])#[:num_instances])
                evals = torch.Tensor(dictionary['evals'])#[:num_instances])

            # Load global information
            if self.load_global_info == True:
                num_nodes = example.x.size(0)
                global_info = torch.Tensor(np.concatenate([[num_nodes], evals]))
                global_info = torch.cat([global_info.unsqueeze(dim = 0) for i in range(num_nodes)], dim = 0)
                example.x = torch.cat([example.x, global_info], dim = 1)

             # Load persistence diagram and neighbor list

            if pl:
                first_index = graph_index
                neighbor_f = 'node_neighbors_pl/'
            else:
                neighbor_f = 'node_neighbors/'

            if self.load_pd == True:
                file_name = '/home/zluo' + '/' + neighbor_f + str(first_index) + '.node_neighbor_features.pkl'
                f = open(file_name, 'rb')
                dictionary = pickle.load(f)
                f.close()

                pd = torch.Tensor(dictionary['pd'])
                neighbor_list = torch.Tensor(dictionary['neighbor'])

                assert pd.size(0) == num_instances
                assert neighbor_list.size(0) == num_instances
                
                example.x = torch.cat([example.x, pd, neighbor_list], dim = 1)
                
            else:
                file_name = data_dir + '/' + neighbor_f + str(first_index) + '.node_neighbor_features.pkl'
                f = open(file_name, 'rb')
                dictionary = pickle.load(f)
                f.close()

                pd = torch.Tensor(dictionary['pd'])
                neighbor_list = torch.Tensor(dictionary['neighbor'])

                assert pd.size(0) == num_instances
                assert neighbor_list.size(0) == num_instances

                example.x = torch.cat([example.x, neighbor_list], dim = 1)

            if concat:
                node_feat = example.x
                net_feat = example.x_net
                fill_node_feat = torch.cat([node_feat, torch.zeros(node_feat.size(0), net_feat.size(1))], dim=1)
                fill_net_feat = torch.cat([torch.zeros(net_feat.size(0), node_feat.size(1)), net_feat], dim=1)
                node_feat = torch.cat([fill_node_feat, fill_net_feat], dim=0)
                example.x = node_feat
                example.x_net = None
            
            self.data.append(example)
        print('Done reading data')

    def len(self):
        return self.num_samples

    def get(self, idx):
        return self.data[idx]

