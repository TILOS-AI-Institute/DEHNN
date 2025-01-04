#!/bin/bash
program=train_gnn_hetero_net

target=$4
#target=capacity
#target=congestion
#target=classify

mkdir $program
cd $program
mkdir $target
cd ..
dir=./$program/$target/

data_dir=../../data/2023-03-06_data/

num_epoch=500
batch_size=1
learning_rate=0.001
seed=123456789
n_layers=$1
hidden_dim=64
design=19
pl=0

# Device
device=cuda
device_idx=$5

# Position encoding
pe=lap
pos_dim=10

# Global information
load_global_info=0
# Persistence diagram & Neighbor list
load_pd=$2

# Test mode
test_mode=0

# GNN type
gnn_type=$3

# Virtual node
virtual_node=$6

for fold in 0
do
name=${program}.${target}.num_epoch.${num_epoch}.batch_size.${batch_size}.learning_rate.${learning_rate}.seed.${seed}.n_layers.${n_layers}.hidden_dim.${hidden_dim}.pe.${pe}.pos_dim.${pos_dim}.virtual_node.${virtual_node}.gnn_type.${gnn_type}.load_global_info.${load_global_info}.load_pd.${load_pd}.fold.${fold}.design.${design}.pl.${pl}
CUDA_VISIBLE_DEVICES=$device_idx python3 $program.py --target=$target --dir=$dir --name=$name --num_epoch=$num_epoch --batch_size=$batch_size --learning_rate=$learning_rate --seed=$seed --n_layers=$n_layers --hidden_dim=$hidden_dim --pe=$pe --pos_dim=$pos_dim --virtual_node=$virtual_node --gnn_type=$gnn_type --load_global_info=$load_global_info --load_pd=$load_pd --test_mode=$test_mode --device=$device --data_dir=$data_dir --fold=$fold --design=$design --pl=$pl
done
