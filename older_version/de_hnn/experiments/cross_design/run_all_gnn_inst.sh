#!/bin/bash

#source train_gnn_hetero.sh 4 1 gcn classify 1 0

#source train_gnn_hetero.sh 4 1 gat classify 1 0

source train_gnn_hetero.sh 4 1 hmpnn classify 2 0

source train_gnn_hetero.sh 4 1 hnhn classify 2 0

source train_gnn_hetero.sh 4 1 allset classify 2 0

#source train_gnn_hetero_base.sh 0 1 gcn classify 2 0

#source train_gnn_hetero_base.sh 0 1 sage classify 0 0

#hyper_no_dir
#source train_gnn_hetero.sh 0 1 hypernodir classify 0 0

#hyper normal
#source train_gnn_hetero.sh 0 1 gat classify 2 0

#hyper VN
#source train_gnn_hetero.sh 0 1 gat classify 2 0


#source train_gnn_hetero.sh 0 1 sota classify 2 0

