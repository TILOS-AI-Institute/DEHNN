#!/bin/bash

for n in 3 8 13 24 30 36 42 47 52 58 64 70; 
do 
	source train_netlistgnn_net.sh $n hpwl 1
done

#for n in 3 8 13 24 30 36 42 47 52 58 64 70; 
#do 
# 	source train_netlistgnn_net.sh $n demand 1
#done
