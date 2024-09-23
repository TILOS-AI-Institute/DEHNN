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
    def __init__(self, data_dir, load_pe = True, num_eigen = 5, pl = True, processed = False, load_indices = None, density = False):
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
            if os.path.exists(data_load_fp):
                data = torch.load(data_load_fp)
            else:
                try:
                    with open(os.path.join(data_dir, design_fp, 'net2sink_nodes.pkl'), 'rb') as f:
                        net2sink = pickle.load(f)
        
                    with open(os.path.join(data_dir, design_fp, 'net2source_node.pkl'), 'rb') as f:
                        net2source = pickle.load(f)
        
                    with open(os.path.join(data_dir, design_fp, 'node_type_id.pkl'), 'rb') as f:
                        node_type = pickle.load(f)
                    
                    # with open(os.path.join(data_dir, design_fp, 'node_loc_x.pkl'), 'rb') as f:
                    #     node_loc_x = pickle.load(f)
                    
                    # with open(os.path.join(data_dir, design_fp, 'node_loc_y.pkl'), 'rb') as f:
                    #     node_loc_y = pickle.load(f)
                    
                    with open(os.path.join(data_dir, design_fp, 'node_size_x.pkl'), 'rb') as f:
                        node_size_x = pickle.load(f)
                    
                    with open(os.path.join(data_dir, design_fp, 'node_size_y.pkl'), 'rb') as f:
                        node_size_y = pickle.load(f)
        
                    # with open(os.path.join(data_dir, design_fp, 'target_net_hpwl.pkl'), 'rb') as f:
                    #     net_hpwl = pickle.load(f)
                    
                    # with open(os.path.join(data_dir, design_fp, 'target_node_congestion_level.pkl'), 'rb') as f:
                    #     node_congestion = pickle.load(f)
        
                    with open(os.path.join(data_dir, design_fp, 'eigen.5.pkl'), 'rb') as f:
                        eig_dict = pickle.load(f)
                        eig_vec = eig_dict['evects']

                    with open(os.path.join(data_dir, design_fp, 'random.pkl'), 'rb') as f:
                        random_dict = pickle.load(f)
                        
                        node_ramdom_features = random_dict['node_random']
                        net_random_features = random_dict['net_random']
        
                    num_instances = len(node_type)
                    # assert len(node_loc_x) == num_instances
                    # assert len(node_loc_y) == num_instances
                    assert len(node_size_x) == num_instances
                    assert len(node_size_y) == num_instances
                    assert len(eig_vec) == num_instances
                    #assert len(node_congestion) == num_instances
        
                    edge_index_source_sink = []
                    #edge_index_sink_source = []
                    edge_index_sink_to_net = []
                    edge_index_source_to_net = []
                    
                    for net_idx in range(len(net2sink)):
                        sink_idx_lst = net2sink[net_idx]
                        source_idx = net2source[net_idx]
                    
                        for sink_idx in sink_idx_lst:
                            edge_index_sink_to_net.append([sink_idx, net_idx])
                            edge_index_source_sink.append([source_idx, sink_idx])
                            #edge_index_sink_source.append([sink_idx, source_idx])
                    
                        edge_index_source_to_net.append([source_idx, net_idx])
                        
        
                    edge_index_source_sink = torch.tensor(edge_index_source_sink).T.long()
                    #edge_index_sink_source = torch.tensor(edge_index_sink_source).T.long()
                    edge_index_source_to_net = torch.tensor(edge_index_source_to_net).T.long()
                    edge_index_sink_to_net = torch.tensor(edge_index_sink_to_net).T.long()
                    out_degrees = compute_degrees(edge_index_source_sink, num_instances)
                    in_degrees = compute_degrees(torch.flip(edge_index_source_sink, dims=[0]), num_instances)
                    source2net_inst_degrees = compute_degrees(edge_index_source_to_net, num_instances)
                    #sink2net_inst_degrees = compute_degrees(edge_index_sink_to_net, num_instances)
                    source2net_net_degrees = compute_degrees(torch.flip(edge_index_source_to_net, dims=[0]), len(net2source))
                    sink2net_net_degrees = compute_degrees(torch.flip(edge_index_sink_to_net, dims=[0]), len(net2source))
                            
                    # if pl:
                    #     node_features = np.vstack([node_type, in_degrees, out_degrees, source2net_inst_degrees, node_size_x, node_size_y, node_loc_x, node_loc_y]).T  
                    #     file_name = os.path.join(data_dir, design_fp, 'pl_part_dict.pkl')
                    #     f = open(file_name, 'rb')
                    #     part_dict = pickle.load(f)
                    #     f.close()
                    #     batch = [part_dict[idx] for idx in range(node_features.shape[0])]
                    #     num_vn = len(np.unique(batch))
                    #     batch = torch.tensor(batch).long()
                    #else:
                    
                    node_features = np.vstack([node_type, in_degrees, out_degrees, source2net_inst_degrees, node_size_x, node_size_y]).T
                    batch = None
                    num_vn = 0
    
                    if load_pe:
                        node_features = np.concatenate([node_features, eig_vec], axis=1)
                        
                    # if density:
                    #     node_route_utilization = np.zeros(num_instances)
                    #     node_pin_utilization = np.zeros(num_instances)
                    #     with open(os.path.join(data_dir, design_fp, "pin_utilization_map.pkl"), 'rb') as f:
                    #         pin_utilization_map = pickle.load(f)
                    #     with open(os.path.join(data_dir, design_fp, "route_utilization_map.pkl"), 'rb') as f:
                    #         route_utilization_map = pickle.load(f)
    
                    #     for node_idx in range(num_instances):
                    #         node_pin_utilization[node_idx] = pin_utilization_map[int(node_loc_x[node_idx]), int(node_loc_y[node_idx])]
                    #         node_route_utilization[node_idx] = route_utilization_map[int(node_loc_x[node_idx]), int(node_loc_y[node_idx])]
    
                    #     node_pin_utilization = node_pin_utilization.reshape(num_instances, 1)
                    #     node_route_utilization = node_route_utilization.reshape(num_instances, 1)
                    #     node_features = np.concatenate([node_features, node_pin_utilization, node_route_utilization], axis=1)
                            
                    node_features = torch.tensor(node_features).float()
                    node_features = torch.concat([node_features, node_ramdom_features], dim=1)
                    net_features = torch.tensor(np.vstack([source2net_net_degrees, sink2net_net_degrees]).T).float()
                    net_features = torch.concat([net_features, net_random_features], dim=1)
                        
                    # file_name = os.path.join(data_dir, design_fp, 'split.pkl')
                    # f = open(file_name, 'rb')
                    # dictionary = pickle.load(f)
                    # f.close()
            
                    # train_indices = dictionary['train_indices']
                    # valid_indices = dictionary['valid_indices']
                    # test_indices = dictionary['test_indices']
    
                    # net_train_indices = dictionary['net_train_indices']
                    # net_valid_indices = dictionary['net_valid_indices']
                    # net_test_indices = dictionary['net_test_indices']
        
                    data = Data(
                        node_features = node_features, 
                        net_features = net_features, 
                        edge_index_source_sink = edge_index_source_sink,
                        edge_index_sink_to_net = edge_index_sink_to_net, 
                        edge_index_source_to_net = edge_index_source_to_net, 
                        #node_congestion = torch.tensor(node_congestion).long(), 
                        #net_hpwl = torch.tensor(net_hpwl).float(),
                        #batch = batch
                    )
                    
                    data_save_fp = os.path.join(data_dir, design_fp, 'pyg_data.pkl')
                    torch.save(data, data_save_fp)
                except:
                    #print(f"failed reading {design_fp}")
                    continue

            data['design_name'] = design_fp
            self.data_lst.append(data)

            
    def len(self):
        return len(self.data_lst)

    def get(self, idx):
        return self.data_lst[idx]
