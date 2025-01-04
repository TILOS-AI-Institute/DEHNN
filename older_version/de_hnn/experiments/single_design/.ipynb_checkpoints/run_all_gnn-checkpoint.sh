#!/bin/bash

for n in 3 8 13 24 30 36 42 47 52 58 64 70; #3 8 47
do	
	for split in 1 2 3;
	do
		source train_gnn_hetero.sh classify gcn 4 1 0 $split $n 0
		source train_gnn_hetero.sh classify gat 4 1 0 $split $n 0
		source train_gnn_hetero.sh classify hyper 3 0 0 $split $n 0
		source train_gnn_hetero.sh classify hyper 3 1 1 $split $n 0
		#source train_netlistgnn_net.sh $n demand 1
	done
done

#for n in 8 30 36 52 58;
#do
#	for split in 1 2 3 4;
#	do
#		#source train_gnn_hetero.sh hpwl hyper 3 1 1 $split $n 2
		#source train_gnn_hetero.sh hpwl gcn 4 1 0 $split $n 2
		#source train_gnn_hetero.sh hpwl gat 4 1 0 $split $n 2
		#source train_gnn_hetero.sh hpwl hyper 3 0 0 $split $n 2
#		source train_netlistgnn_net.sh $n demand 1
#	done
#done




