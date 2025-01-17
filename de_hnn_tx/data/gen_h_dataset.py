import torch
from torch_geometric.utils import scatter
from torch_geometric.data import HeteroData
import numpy as np
from tqdm import tqdm
import os
import pickle

def process_vlsi_dataset(dataset, target_data_dir):
    """
    Process VLSI dataset to create heterogeneous graphs with layout information.
    
    Args:
        dataset: Input dataset containing VLSI design information
        target_data_dir: Directory containing target layout data
    
    Returns:
        List[HeteroData]: Processed heterogeneous graph dataset
    """
    def filter_edges(data, degree_threshold=3000):
        out_degrees = data.net_features[:, 1]
        mask = (out_degrees < degree_threshold)
        
        # Filter source-to-net edges
        mask_source = mask[data.edge_index_source_to_net[1]]
        data.edge_index_source_to_net = data.edge_index_source_to_net[:, mask_source]
        
        # Filter sink-to-net edges
        mask_sink = mask[data.edge_index_sink_to_net[1]]
        data.edge_index_sink_to_net = data.edge_index_sink_to_net[:, mask_sink]
        
        return data

    def load_layout_data(design_fp, target_data_dir):
        base_path = os.path.join(target_data_dir, design_fp)
        layout_data = {}
        
        for file_name, key in [
            ('node_loc_x.pkl', 'x'),
            ('node_loc_y.pkl', 'y'),
            ('target_net_hpwl.pkl', 'hpwl'),
            ('target_node_utilization.pkl', 'congestion')
        ]:
            with open(os.path.join(base_path, file_name), 'rb') as f:
                layout_data[key] = pickle.load(f)
        
        return layout_data

    def process_variant_data(data, pos_lst, node_congestion, net_hpwl):
        edge_index = torch.concat([data.edge_index_source_to_net, data.edge_index_sink_to_net], dim=1)
        
        # Calculate edge attributes
        node_pos_lst = pos_lst[data.edge_index_sink_to_net[0]]
        net_pos_lst = pos_lst[data.edge_index_source_to_net[0]][data.edge_index_sink_to_net[1]]
        edge_attr = torch.sum(torch.abs(node_pos_lst - net_pos_lst), dim=1)
        
        # Process node positions for nets
        source_nodes = edge_index[0]
        target_nodes = edge_index[1]
        values_per_edge = pos_lst[source_nodes]
        pos_lst_net = scatter(values_per_edge, target_nodes, dim=0, reduce="mean")
        
        return pos_lst, pos_lst_net, edge_attr, node_congestion/580, net_hpwl

    h_dataset = []
    for data in tqdm(dataset, desc="Processing designs"):
        # Filter edges based on degree threshold
        data = filter_edges(data)
        
        # Create heterogeneous graph
        h_data = HeteroData()
        h_data['node'].node_features = data.node_features
        h_data['net'].net_features = data.net_features
        h_data.num_instances = data.node_features.shape[0]
        h_data['node', 'as_a_sink_of', 'net'].edge_index = data.edge_index_sink_to_net
        h_data['node', 'as_a_source_of', 'net'].edge_index = data.edge_index_source_to_net
        
        # Process variants
        design_num = data['design_name'].split("_")[1]
        variant_data_lst = []
        
        for design_fp in os.listdir(target_data_dir):
            if design_num == design_fp.split("_")[1]:
                layout_data = load_layout_data(design_fp, target_data_dir)
                
                pos_lst = torch.tensor(np.vstack([layout_data['x'], layout_data['y']]).T)
                assert pos_lst.shape[0] == h_data['node'].node_features.shape[0]
                
                variant_data = process_variant_data(
                    data,
                    pos_lst,
                    torch.tensor(layout_data['congestion']).float(),
                    torch.tensor(layout_data['hpwl']).float()
                )
                variant_data_lst.append(variant_data)
        
        h_data['variant_data_lst'] = variant_data_lst
        h_dataset.append(h_data)
    
    return h_dataset