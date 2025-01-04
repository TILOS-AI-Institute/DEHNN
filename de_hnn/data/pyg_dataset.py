import os
import torch
import torch.nn
from torch_geometric.data import Dataset
from torch_geometric.data import Data
from tqdm import tqdm
import pickle
import numpy as np

from utils import *

class NetlistDataset(Dataset):
    def __init__(self, data_dir, load_pe = True, load_pd = True, num_eigen = 10, pl = True, processed = False, load_indices = None, density = False):
        super().__init__()
        self.data_dir = data_dir
        self.data_lst = []

        all_files = np.array(os.listdir(data_dir))
        
        if load_indices is not None:
            load_indices = np.array(load_indices)
            all_files = all_files[load_indices]
            
        for design_fp in tqdm(all_files):
            #print(design_fp)
            data_load_fp = os.path.join(data_dir, design_fp, 'pyg_data.pkl')
            if processed and os.path.exists(data_load_fp):
                data = torch.load(data_load_fp)
            else:
                data_load_fp = os.path.join(data_dir, design_fp)

                file_name = data_load_fp + '/' + 'node_features.pkl'
                f = open(file_name, 'rb')
                dictionary = pickle.load(f)
                f.close()        
                self.design_name = dictionary['design']
                num_instances = dictionary['num_instances']
                num_nets = dictionary['num_nets']
                raw_instance_features = torch.Tensor(dictionary['instance_features'])
                pos_lst = raw_instance_features[:, :2]

                x_min = dictionary['x_min']
                x_max = dictionary['x_max']
                y_min = dictionary['y_min']
                y_max = dictionary['y_max'] 
                min_cell_width = dictionary['min_cell_width']
                max_cell_width = dictionary['max_cell_width']
                min_cell_height = dictionary['min_cell_height']
                max_cell_height = dictionary['max_cell_height']
                
                X = pos_lst[:, 0].flatten()
                Y = pos_lst[:, 1].flatten()
                
                instance_features = raw_instance_features[:, 2:]
                net_features = torch.zeros(num_nets, instance_features.size(1))
                
                file_name = data_load_fp + '/' + 'bipartite.pkl'
                f = open(file_name, 'rb')
                dictionary = pickle.load(f)
                f.close()
        
                instance_idx = torch.Tensor(dictionary['instance_idx']).unsqueeze(dim = 1).long()
                net_idx = torch.Tensor(dictionary['net_idx']) + num_instances
                net_idx = net_idx.unsqueeze(dim = 1).long()
                
                edge_attr = torch.Tensor(dictionary['edge_attr']).float().unsqueeze(dim = 1).float()
                edge_index = torch.cat((instance_idx, net_idx), dim = 1)
                edge_dir = dictionary['edge_dir']
                v_drive_idx = [idx for idx in range(len(edge_dir)) if edge_dir[idx] == 1]
                v_sink_idx = [idx for idx in range(len(edge_dir)) if edge_dir[idx] == 0] 
                edge_index_source_to_net = edge_index[v_drive_idx]
                edge_index_sink_to_net = edge_index[v_sink_idx]
                
                edge_index_source_to_net = torch.transpose(edge_index_source_to_net, 0, 1)
                edge_index_sink_to_net = torch.transpose(edge_index_sink_to_net, 0, 1)
                
                x = instance_features
                
                example = Data()
                example.__num_nodes__ = x.size(0)
                example.x = x

                fn = data_load_fp + '/' + 'degree.pkl'
                f = open(fn, "rb")
                d = pickle.load(f)
                f.close()

                example.edge_attr = edge_attr[:2]
                example.cell_degrees = torch.tensor(d['cell_degrees'])
                example.net_degrees = torch.tensor(d['net_degrees'])
                
                example.x = torch.cat([example.x, example.cell_degrees.unsqueeze(dim = 1)], dim = 1)
                example.x_net = example.net_degrees.unsqueeze(dim = 1)

                file_name = data_load_fp + '/' + 'metis_part_dict.pkl'
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

                file_name = data_dir + '/' + str(sample) + '.net_demand_capacity.pkl'
                f = open(file_name, 'rb')
                net_demand_dictionary = pickle.load(f)
                f.close()

                net_demand = torch.Tensor(net_demand_dictionary['demand'])

                file_name = data_dir + '/' + str(sample) + '.targets.pkl'
                f = open(file_name, 'rb')
                node_demand_dictionary = pickle.load(f)
                f.close()

                node_demand = torch.Tensor(node_demand_dictionary['demand'])
                
                fn = data_load_fp + '/' + 'net_hpwl.pkl'
                f = open(fn, "rb")
                d_hpwl = pickle.load(f)
                f.close()
                net_hpwl = torch.Tensor(d_hpwl['hpwl']).float()


                if load_pe:
                    file_name = data_load_fp + '/' + 'eigen.10.pkl'
                    f = open(file_name, 'rb')
                    dictionary = pickle.load(f)
                    f.close()

                    evects = torch.Tensor(dictionary['evects'])
                    example.x = torch.cat([example.x, evects[:example.x.shape[0]]], dim = 1)
                    example.x_net = torch.cat([example.x_net, evects[example.x.shape[0]:]], dim = 1)

                if load_pd == True:
                    file_name = data_load_fp + '/' + 'node_neighbor_features.pkl'
                    f = open(file_name, 'rb')
                    dictionary = pickle.load(f)
                    f.close()
        
                    pd = torch.Tensor(dictionary['pd'])
                    neighbor_list = torch.Tensor(dictionary['neighbor'])
        
                    assert pd.size(0) == num_instances
                    assert neighbor_list.size(0) == num_instances
        
                    example.x = torch.cat([example.x, pd, neighbor_list], dim = 1)

                data = Data(
                        node_features = example.x, 
                        net_features = example.x_net, 
                        edge_index_sink_to_net = edge_index_sink_to_net, 
                        edge_index_source_to_net = edge_index_source_to_net, 
                        node_demand = node_demand, 
                        net_demand = net_demand,
                        net_hpwl = net_hpwl,
                        batch = example.part_id,
                        num_vn = example.num_vn,
                        pos_lst = pos_lst
                    )
                
                data_save_fp = os.path.join(data_load_fp, 'pyg_data.pkl')
                torch.save(data, data_save_fp)
                    

            data['design_name'] = design_fp
            self.data_lst.append(data)

            
    def len(self):
        return len(self.data_lst)

    def get(self, idx):
        return self.data_lst[idx]
