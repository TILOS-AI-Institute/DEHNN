#!/bin/bash

source train_gnn_hetero_net.sh 4 1 gcn hpwl 1 0

source train_gnn_hetero_net.sh 4 1 gat hpwl 1 0

#source train_gnn_hetero_net.sh 4 1 sota demand 1 0

#source train_gnn_hetero_net.sh 0 1 gat demand 1 0

#source train_gnn_hetero_net.sh 0 1 gat demand 1 0

#source train_netlistgnn_net.sh 0 demand 1

#gcn
#source train_gnn_hetero_demand.sh 0 1 gcn demand 4 0

#gat
#source train_gnn_hetero_demand.sh 0 1 sota demand 2 0

source train_gnn_hetero_net.sh 3 0 hyper hpwl 1 0

source train_gnn_hetero_net.sh 3 1 hyper hpwl 1 0

source train_gnn_hetero_net.sh 3 1 hyper hpwl 1 1

# # #hyper normal
#source train_gnn_hetero_net.sh 3 1 hyper demand 2 0

 #hyper VN
#source train_gnn_hetero_net.sh 3 0 hyper demand 2 0


#source train_gnn_hetero_demand.sh 0 1 hyper demand 6 2
