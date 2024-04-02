#!/bin/bash

program=train_mlp_net


target=$2
#target=demand
#target=capacity
#target=congestion

# Index of graph
graph_index=$1

mkdir $program
cd $program
mkdir $target
cd ..
dir=./$program/$target/

data_dir=../../data/2023-03-06_data/

num_epoch=1000
batch_size=1
learning_rate=0.001
seed=123456789
hidden_dim=64

# Device
device=cuda
device_idx=7

# Position encoding
pe=lap
pos_dim=10

# Global information
load_global_info=0

# Persistence diagram & Neighbor list
load_pd=0

# Test mode
test_mode=0

# Execution
name=${program}.${target}.num_epoch.${num_epoch}.batch_size.${batch_size}.learning_rate.${learning_rate}.seed.${seed}.hidden_dim.${hidden_dim}.pe.${pe}.pos_dim.${pos_dim}.load_global_info.${load_global_info}.load_pd.${load_pd}.graph_index.${graph_index}
CUDA_VISIBLE_DEVICES=$device_idx python3 $program.py --target=$target --dir=$dir --name=$name --num_epoch=$num_epoch --batch_size=$batch_size --learning_rate=$learning_rate --seed=$seed --hidden_dim=$hidden_dim --pe=$pe --pos_dim=$pos_dim --load_global_info=$load_global_info --load_pd=$load_pd --test_mode=$test_mode --device=$device --data_dir=$data_dir --graph_index=$graph_index


