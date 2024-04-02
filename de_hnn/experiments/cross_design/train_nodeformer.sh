#!/bin/bash

program=train_nodeformer

#target=demand
#target=capacity
target=congestion

mkdir $program
cd $program
mkdir $target
cd ..
dir=./$program/$target/

data_dir=../../data/2023-03-06_data/

num_epoch=100
batch_size=1
learning_rate=0.001
seed=123456789
n_layers=1
hidden_dim=10

# Device
device=cuda
device_idx=5

# Position encoding
pe=none
pos_dim=10

# Global information
load_global_info=1

# Persistence diagram & Neighbor list
load_pd=1

# Test mode
test_mode=0

# NodeFormer
rb_order=0

for fold in 0 1 2 3 4 5
do
name=${program}.target.${target}.rb_order.${rb_order}.num_epoch.${num_epoch}.batch_size.${batch_size}.learning_rate.${learning_rate}.seed.${seed}.n_layers.${n_layers}.hidden_dim.${hidden_dim}.pe.${pe}.pos_dim.${pos_dim}.load_global_info.${load_global_info}.load_pd.${load_pd}.fold.${fold}
CUDA_VISIBLE_DEVICES=$device_idx python3 $program.py --target=$target --rb_order=$rb_order --dir=$dir --name=$name --num_epoch=$num_epoch --batch_size=$batch_size --learning_rate=$learning_rate --seed=$seed --n_layers=$n_layers --hidden_dim=$hidden_dim --pe=$pe --pos_dim=$pos_dim --test_mode=$test_mode --device=$device --data_dir=$data_dir --load_global_info=$load_global_info --load_pd=$load_pd --fold=$fold
done

