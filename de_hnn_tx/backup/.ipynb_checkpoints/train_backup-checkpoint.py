    # _, data.edge_weights_sink_to_net = gcn_norm(data.edge_index_sink_to_net)
    # data.edge_index_net_to_node, data.edge_weights_net_to_node = gcn_norm(edge_index_net_to_node)
    # data.edge_weights_source_to_net = torch.ones(data.edge_index_source_to_net.shape[1]).float()

    #mask = torch.tensor([True for idx in range(len(h_data['net'].x))])
    #loader = NeighborLoader(
    #    h_data,
    #    num_neighbors={key: [15] * 4 for key in h_data.edge_types},
    #    input_nodes=('net', mask),
    #    batch_size=160000 
    #)

    #h_dataset.append(loader)

#pickle.dump(h_dataset, open("h_neighbor_loader_lst.pt", "wb"))

