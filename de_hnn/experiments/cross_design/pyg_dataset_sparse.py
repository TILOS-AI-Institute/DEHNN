import torch
import torch.nn
from torch_geometric.data import Dataset
from torch_geometric.data import Data

import numpy as np
import pickle

from tqdm import tqdm

class pyg_dataset(Dataset):
    def __init__(self, data_dir, fold_index, split, target, load_pe = False, num_eigen = 5, load_global_info = True, load_pd = False, design=19, pl=0, graph_rep = 'bipartite', vn=False, net = True, concat = False):
        super().__init__()
        self.data_dir = data_dir
        self.fold_index = fold_index
        self.split = split
        self.target = target
        assert target == 'demand' or target == 'capacity' or target == 'congestion' or target == 'classify' or target == 'hpwl'
        print('Learning target:', self.target)

        # Position encoding
        self.load_pe = load_pe
        self.num_eigen = num_eigen
        self.load_global_info = load_global_info
        self.load_pd = load_pd

        # Read cross-validation
        pl = bool(pl)
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

        # Read data
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
            
            if not pl:
                instance_features = instance_features[:, 2:]
            else:
                #pos_lst = dictionary['instance_features'][:, :2]
                #X, Y = pos_lst.T[0], pos_lst.T[1]
                #xloc_list = X*(dictionary['x_max'] - dictionary['x_min']) + dictionary['x_min']
                #yloc_list = Y*(dictionary['y_max'] - dictionary['y_min']) + dictionary['y_min']
                #instance_features = torch.cat([instance_features[:, 2:], torch.Tensor(xloc_list).unsqueeze(dim = 1), torch.Tensor(yloc_list).unsqueeze(dim = 1)], dim=1) 
                instance_features = instance_features
            # Read learning targets
            if not net:
                file_name = data_dir + '/' + str(graph_index) + '.targets.pkl'
                f = open(file_name, 'rb')
                dictionary = pickle.load(f)
                f.close()
            else:
                file_name = data_dir + '/' + str(graph_index) + '.net_demand_capacity.pkl'
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

            row = dictionary['instance_idx']
            col = dictionary['net_idx']
            edge_dir = dictionary['edge_dir']
            data = torch.ones(len(row))

            i = np.array([row, col])
        
            net_inst_adj = torch.sparse_coo_tensor(i, data).t()
            
            v_drive_idx = [idx for idx in range(len(row)) if edge_dir[idx] == 1]
            v_sink_idx = [idx for idx in range(len(row)) if edge_dir[idx] == 0] 
            
            inst_net_adj_v_drive = torch.sparse_coo_tensor(i.T[v_drive_idx].T, data[v_drive_idx])
            
            inst_net_adj_v_sink = torch.sparse_coo_tensor(i.T[v_sink_idx].T, data[v_sink_idx])
            
            x = instance_features
            
            with open(f"../../data/2023-03-06_data/all.idx_to_design.pkl", "rb") as f:
                idx_dict = pickle.load(f)

            first_index = idx_dict[graph_index]

            with open(f"../../data/2023-03-06_data/{first_index}.nn_conn.pkl", "rb") as f:
                conn_dict = pickle.load(f)

            # PyG data
            
            example = Data()
            example.__num_nodes__ = x.size(0)
            example.x = x
            
#             # Load capacity
            capacity = capacity.unsqueeze(dim = 1)
            norm_cap = (capacity - torch.min(capacity)) / (torch.max(capacity) - torch.min(capacity))
            capacity_features = torch.cat([capacity, torch.sqrt(capacity), norm_cap, torch.sqrt(norm_cap), torch.square(norm_cap), torch.sin(norm_cap), torch.cos(norm_cap)], dim = 1)

#             example.x_net = capacity_features
            example.num_instances = num_instances
            example.y = y
            example.net_inst_adj = net_inst_adj
            example.inst_net_adj_v_drive = inst_net_adj_v_drive
            example.inst_net_adj_v_sink = inst_net_adj_v_sink
           
            fn = data_dir + '/' + str(graph_index) + f'.degree.pkl'
            f = open(fn, "rb")
            d = pickle.load(f)
            f.close()

            example.cell_degrees = torch.Tensor(d['cell_degrees'])
            example.net_degrees = torch.Tensor(d['net_degrees'])
            #example.x = torch.cat([example.x, example.cell_degrees.unsqueeze(dim = 1)], dim = 1)
            
            if (not net) or (not pl):
                example.x_net = example.net_degrees.unsqueeze(dim = 1)
            else:
                example.x_net = capacity_features
                example.x_net = torch.cat([example.x_net, example.net_degrees.unsqueeze(dim = 1)], dim = 1)

            if vn:       
                file_name = data_dir + '/' + str(graph_index) + '.metis_part_dict.pkl'
                if pl:
                    file_name = data_dir + '/' + str(graph_index) + '.pl_part_dict.pkl'
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
                #example.edge_index_local_vn = torch.Tensor(edge_index_local_vn).long().t()
                #example.edge_index_vn_top = torch.Tensor(edge_index_vn_top).long().t()
            
            if pl and (not net):
                file_name = data_dir + '/' + str(graph_index) + '.targets.pkl'
                f = open(file_name, 'rb')
                dictionary = pickle.load(f)
                f.close()
                capacity = torch.Tensor(dictionary['capacity'])
                capacity = capacity.unsqueeze(dim = 1)
                norm_cap = (capacity - torch.min(capacity)) / (torch.max(capacity) - torch.min(capacity))
                capacity_features = torch.cat([capacity, torch.sqrt(capacity), norm_cap, torch.sqrt(norm_cap), torch.square(norm_cap), torch.sin(norm_cap), torch.cos(norm_cap)], dim = 1)
                example.x = torch.cat([example.x, capacity_features], dim = 1)
            
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
                nerighbor_f = 'node_neighbors_pl/'
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
                node_feat = example.x
                net_feat = example.x_net
                fill_node_feat = torch.cat([node_feat, torch.zeros(node_feat.size(0), net_feat.size(1))], dim=1)
                fill_net_feat = torch.cat([torch.zeros(net_feat.size(0), node_feat.size(1)), net_feat], dim=1)
                node_feat = torch.cat([fill_node_feat, fill_net_feat], dim=0)
                example.x = node_feat

            self.data.append(example)

        print('Done reading data')

    def len(self):
        return self.num_samples

    def get(self, idx):
        return self.data[idx]

