import torch
import torch.nn
from torch_geometric.data import Dataset
from torch_geometric.data import Data

import numpy as np
import pickle

class pyg_dataset(Dataset):
    def __init__(self, data_dir, graph_index, target, load_pe = False, num_eigen = 5, load_global_info = True, load_pd = False, graph_rep = 'star', vn = False, net = False, concat = False, split = 1, pl = 0):
        super().__init__()
        self.data_dir = data_dir
        self.graph_index = graph_index
        self.target = target
        assert target == 'demand' or target == 'capacity' or target == 'congestion' or target == 'classify' or target == 'hpwl'
        print('Learning target:', self.target)

        self.load_pe = load_pe
        self.num_eigen = num_eigen
        self.load_global_info = load_global_info
        self.load_pd = load_pd
        

        print(f"Placement Information {pl}")
        split_data_dir = data_dir + f"/split/{split}/"
        print(split_data_dir)
        if net == True:
            file_name = split_data_dir + str(graph_index) + '.split_net.pkl'
            f = open(file_name, 'rb')
            dictionary = pickle.load(f)
            f.close()
            
        else:
            file_name = split_data_dir + str(graph_index) + '.split.pkl'
            f = open(file_name, 'rb')
            dictionary = pickle.load(f)
            f.close()

        self.train_indices = dictionary['train_indices']
        self.valid_indices = dictionary['valid_indices']
        self.test_indices = dictionary['test_indices']

        # Read node features
        file_name = data_dir + '/' + str(graph_index) + '.node_features.pkl'
        f = open(file_name, 'rb')
        dictionary = pickle.load(f)
        f.close()

        self.design_name = dictionary['design']

        num_instances = dictionary['num_instances']
        num_nets = dictionary['num_nets']
        instance_features = torch.Tensor(dictionary['instance_features'])
        #if not pl:
        instance_features = instance_features[:, 2:]
        net_features = torch.zeros(num_nets, instance_features.size(1))

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
        file_name = data_dir + '/' + str(graph_index) + '.bipartite.pkl'
        f = open(file_name, 'rb')
        dictionary = pickle.load(f)
        f.close()
        
        # Read Filter Index file
#         file_name = data_dir + '/' + str(graph_index) + '.select.pkl'
#         f = open(file_name, 'rb')
#         select_indices = pickle.load(f)['select_indicess']
#         f.close()
        
        row = dictionary['instance_idx']#[select_indices]
        col = dictionary['net_idx']#[select_indices]
        edge_dir = dictionary['edge_dir']#[select_indices]
        
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
                
        capacity = capacity.unsqueeze(dim = 1)
        norm_cap = (capacity - torch.min(capacity)) / (torch.max(capacity) - torch.min(capacity))
        capacity_features = torch.cat([capacity, torch.sqrt(capacity), norm_cap, torch.sqrt(norm_cap), torch.square(norm_cap), torch.sin(norm_cap), torch.cos(norm_cap)], dim = 1)
        
        example.y = y
        example.net_inst_adj = net_inst_adj
        example.inst_net_adj_v_drive = inst_net_adj_v_drive
        example.inst_net_adj_v_sink = inst_net_adj_v_sink
        example.num_instances = num_instances
        


        if vn:
            if not pl:       
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
            
            #example.edge_index_local_vn = torch.Tensor(edge_index_local_vn).long().t()
            #example.edge_index_vn_top = torch.Tensor(edge_index_vn_top).long().t()
        
        #demand = torch.Tensor(dictionary['demand'])
        #capacity = torch.Tensor(dictionary['capacity'])

        #capacity = torch.sum(capacity, dim = 1).unsqueeze(dim = 1)
        #norm_cap = (capacity - torch.min(capacity)) / (torch.max(capacity) - torch.min(capacity))
        #capacity_features = torch.cat([capacity, torch.sqrt(capacity), norm_cap, torch.sqrt(norm_cap), torch.square(norm_cap), torch.sin(norm_cap), torch.cos(norm_cap)], dim = 1)

        #example.x = torch.cat([example.x, capacity_features], dim = 1)

        fn = data_dir + '/' + str(graph_index) + f'.degree.pkl'
        f = open(fn, "rb")
        d = pickle.load(f)
        f.close()

        example.cell_degrees = torch.tensor(d['cell_degrees'])
        example.net_degrees = torch.tensor(d['net_degrees'])
        
        #example.x = torch.cat([example.x, example.cell_degrees.unsqueeze(dim = 1)], dim = 1)
        
        if not pl or not net:
            example.x_net = example.net_degrees.unsqueeze(dim = 1)
        else:
            example.x_net = capacity_features
            example.x_net = torch.cat([example.x_net, example.net_degrees.unsqueeze(dim = 1)], dim = 1)


        if not net:
            file_name = data_dir + '/' + str(graph_index) + '.targets.pkl'
            f = open(file_name, 'rb')
            dictionary = pickle.load(f)
            f.close()
            capacity = torch.Tensor(dictionary['capacity'])
            capacity = capacity.unsqueeze(dim = 1)
            norm_cap = (capacity - torch.min(capacity)) / (torch.max(capacity) - torch.min(capacity))
            capacity_features = torch.cat([capacity, torch.sqrt(capacity), norm_cap, torch.sqrt(norm_cap), torch.square(norm_cap), torch.sin(norm_cap), torch.cos(norm_cap)], dim = 1)
            #example.x = torch.cat([example.x, capacity_features], dim = 1)

        # Load positional encoding
        if self.load_pe == True:
            file_name = data_dir + '/' + str(first_index) + '.eigen.' + str(self.num_eigen) + '.pkl'
            f = open(file_name, 'rb')
            dictionary = pickle.load(f)
            f.close()
            
            example.evects = torch.Tensor(dictionary['evects'])
            #example.evals = torch.Tensor(dictionary['evals'])

        # Load global information
        if self.load_global_info == True:
            #file_name = data_dir + '/' + str(graph_index) + '.global_information.pkl'
            #f = open(file_name, 'rb')
            #dictionary = pickle.load(f)
            #f.close()

            #core_util = dictionary['core_utilization']
            #global_info = torch.Tensor(np.array([core_util, np.sqrt(core_util), core_util ** 2, np.cos(core_util), np.sin(core_util)]))
            #num_nodes = example.x.size(0)
            #global_info = torch.cat([global_info.unsqueeze(dim = 0) for i in range(num_nodes)], dim = 0)

            #example.x_net = torch.Tensor(example.net_degrees).unsqueeze(dim=1) 
            pass

        if pl:
            first_index = graph_index
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

        self.example = example
        
        print(example, graph_index)

    def len(self):
        return 1

    def get(self, idx):
        return self.example

