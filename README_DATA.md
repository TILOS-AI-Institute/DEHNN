This is the README for netlist dataset we used in our paper. 

Raw Data:
The netlist dataset consists of 12 of the Superblue circuits from (Viswanathan et al., 2011, 2012), including Superblue1,2,3,5,6,7,9,11,14,16,18 and 19. The size of these netlists range from 400K
to 1.3M nodes, with similar number of nets. More details of designs can be found in paper and appendix.

Processed Data:
We used PyTorch-geometric (pyg) (Fey and Lenssen, 2019) to construct the dataset and data objects. 
Depending on the models is a Graph Neural Network or a (directed) Hypergraph Neural Network, each netlist circuit from the Raw Data will be represented as a bipartite-graph or (directed) hypergraph using pyg. 
  
  Features:
    -
    -
    -

  Targets:
    - Net-based Wirelength Regression: Half-perimeter wirelength (HPWL) as a common estimate of wirelength. 
    - Net-based Demand Regression: Demand of each net, congestion happens when demand exceeds capacity. 
    - Cell-based Congestion Classification: Similar to (Yang et al., 2022) and (Wang et al., 2022), we classify the cell-based congestion values (computed as the ratio of cell demand/cell capacity) into (a) [0,0.9], not-congested ; and (b) [0.9, inf]; congested.

(this file is still under active editing)
                                                          
