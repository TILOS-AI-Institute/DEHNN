#!/bin/bash

program=train_transformer
target=classify
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
heads=1
local_heads=1
depth=4

# Global information
load_global_info=0

# Persistence diagram & Neighbor list
load_pd=1

# Device
device=cuda
device_idx=2

# Test mode
test_mode=0

# Position encoding dimension if used
pe_dim=10

# Position encoding
pe_type=lap

# Execution
name=${program}.target.${target}.num_epoch.${num_epoch}.batch_size.${batch_size}.learning_rate.${learning_rate}.seed.${seed}.hidden_dim.${hidden_dim}.heads.${heads}.local_heads.${local_heads}.depth.${depth}.pe_type.${pe_type}.pe_dim.${pe_dim}.load_global_info.${load_global_info}.load_pd.${load_pd}.graph_index.${graph_index}
CUDA_VISIBLE_DEVICES=$device_idx python3 $program.py --target=$target --dir=$dir --data_dir=${data_dir} --name=$name --num_epoch=$num_epoch --batch_size=$batch_size --learning_rate=$learning_rate --seed=$seed --hidden_dim=$hidden_dim --heads=$heads --local_heads=$local_heads --depth=$depth --test_mode=$test_mode --pe_type=$pe_type --pe_dim=$pe_dim --load_global_info=$load_global_info --load_pd=$load_pd --graph_index=$graph_index --device=$device

