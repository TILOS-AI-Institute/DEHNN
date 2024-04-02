PyTorch Geometric (PyG) models are in ```gnn.py``` with the graph convolution implemented in ```graph_conv.py```. We can also use the position encodings from GraphGPS (check ```position_enc/``` for more options).

The baseline of Linear Transformer + LapPE (Laplacian position encoding) is in ```linear_transformer_lappe.py```.

Directory gnn_to_exchange contains two other implementations, one is transfer directed graph to undirected graph, another one implemented directed gcn with two seperate message passing for (source to target) and (target to source). Current reported results coming from the directed gcn version. 
