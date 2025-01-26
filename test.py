import os
import numpy as np
import pickle
import torch
import torch.nn
from torch_geometric.data import Dataset
from torch_geometric.data import Data, HeteroData
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from utils import *
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import matplotlib.pyplot as plt
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool

import sys
sys.path.insert(1, 'data/')
from pyg_dataset import NetlistDataset
from gen_h_dataset import process_vlsi_dataset
from data.utils import compute_degrees

sys.path.append("models/")
sys.path.append("models/layers/")
from models.model import GNN_node
from torch_geometric.utils import scatter

import logging
from typing import Dict, Tuple


def setup_logging():
    logging.basicConfig(
        filename='evaluation.txt',
        level=logging.INFO,
        format='%(asctime)s - %(message)s'
    )

def evaluate_model(model, h_dataset, device, config: Dict) -> Tuple[float, float, float, float]:
    model.eval()
    total_nrmse = total_ssim = total_mape = total_r2 = 0
    total_node_nrmse = total_node_ssim = total_node_mape = total_node_r2 = 0
    num_place = 0
    
    with torch.no_grad():
        for data_idx in tqdm(range(len(h_dataset))):
            data = h_dataset[data_idx]
            data['node', 'as_a_sink_of', 'net'].edge_index = data['node', 'as_a_sink_of', 'net'].edge_index.to(device)
            data['node', 'as_a_source_of', 'net'].edge_index = data['node', 'as_a_source_of', 'net'].edge_index.to(device)
            
            for inner_data_idx in range(len(data.variant_data_lst)):
                pos_lst, pos_lst_net, edge_attr, target_node, target_net = data.variant_data_lst[inner_data_idx]
                if (torch.sum(target_node) < 10) or (torch.sum(target_net) < 10):
                    continue
                
                data['node'].x = torch.concat([data['node'].node_features, pos_lst], dim=1)
                data['net'].x = torch.concat([data['net'].net_features, pos_lst_net], dim=1)
                data['node', 'as_a_sink_of', 'net'].edge_attr = edge_attr
                data.pos_lst = pos_lst
                data.num_sites_x = config['num_sites_x']
                data.num_sites_y = config['num_sites_y']
                target_node = target_node.to(device)
            
                node_representation, net_representation, vn_representation = model(data, device)
                target_map = convert_to_util_map(pos_lst, target_node, config['num_sites_x'], 
                                               config['num_sites_y'], device, "amax").flatten()
                
                node_map = vn_representation.squeeze(1)
                total_nrmse += nrmse(target_map, node_map).item()
                total_ssim += ssim(target_map, node_map).item()
                total_mape += mape(target_map, node_map).item()
                total_r2 += r2_score(target_map, node_map).item()

                total_node_nrmse += nrmse(target_node, node_representation.flatten()).item()
                total_node_ssim += ssim(target_node, node_representation.flatten()).item()
                total_node_mape += mape(target_node, node_representation.flatten()).item()
                total_node_r2 += r2_score(target_node, node_representation.flatten()).item()
        
                num_place += 1
                
                if (total_nrmse > 100000) or (total_r2 < 0):
                    raise ValueError("Invalid metrics detected")

    metrics = {
        'map': (total_nrmse/num_place, total_ssim/num_place, 
                total_mape/num_place, total_r2/num_place),
        'node': (total_node_nrmse/num_place, total_node_ssim/num_place, 
                 total_node_mape/num_place, total_node_r2/num_place)
    }
    return metrics, num_place

def main():
    setup_logging()
    device = "cuda"
    num_sites_x, num_sites_y = 206, 300

    num_layer = 2
    num_dim = 32
    aggr = "add"
    device = "cuda"
    
    learning_rate = 0.0005
    
    configs = [
        #{'vn': False, 'cv': False, 'model_type': 'dehnn'},
        #{'vn': True, 'cv': False, 'model_type': 'dehnn'},
        #{'vn': True, 'cv': True, 'model_type': 'dehnn'},
        {'vn': True, 'cv': True, 'model_type': 'unet'}
    ]
    
    load_indices = np.array(['221', '181', '226', '206', '191', '190', '192', '182', '222', '197', '71', '81', '151', '161', '106', '160', '112', '75', '37', '82', '21', '45', '102', '140', '7'])
    all_load_indices = []
    for file in os.listdir("data/all_designs_netlist_data/"):
        index = file.split("_")[1]
        all_load_indices.append(index)

    test_load_indices = []
    for index in all_load_indices:
        if index not in load_indices:
            test_load_indices.append(index)
    
    test_load_indices = np.array(test_load_indices)
    load_indices = test_load_indices
    test = False # if only test but not train
    restart = True # if restart training
    reload_dataset = False # if reload already processed h_dataset
    target_data_dir = "data/target_data"
    
    dataset = NetlistDataset(data_dir="data/all_designs_netlist_data", 
                            load_pe=True, pl=True, processed=True, 
                            load_indices=load_indices)
    h_dataset = process_vlsi_dataset(dataset, target_data_dir)
    
    for config in configs:
        try:
            model_path = f"{config['model_type']}_{num_layer}_{num_dim}_{config['vn']}_{config['cv']}_model.pt"
            if not os.path.exists(model_path):
                logging.warning(f"Model not found: {model_path}")
                continue
                
            model = torch.load(model_path).to(device)
            logging.info(f"\nEvaluating configuration: {config}")
            
            metrics, num_place = evaluate_model(model, h_dataset, device, 
                                             {'num_sites_x': num_sites_x, 
                                              'num_sites_y': num_sites_y})
            
            logging.info(f"Number of placements evaluated: {num_place}")
            logging.info("Map Metrics:")
            logging.info(f"NRMSE: {metrics['map'][0]:.3f}")
            logging.info(f"SSIM: {metrics['map'][1]:.3f}")
            logging.info(f"MAPE: {metrics['map'][2]:.3f}")
            logging.info(f"R2: {metrics['map'][3]:.3f}")
            
            logging.info("Node Metrics:")
            logging.info(f"NRMSE: {metrics['node'][0]:.3f}")
            logging.info(f"SSIM: {metrics['node'][1]:.3f}")
            logging.info(f"MAPE: {metrics['node'][2]:.3f}")
            logging.info(f"R2: {metrics['node'][3]:.3f}")
            
        except Exception as e:
            logging.error(f"Error evaluating config {config}: {str(e)}")

if __name__ == "__main__":
    main()
