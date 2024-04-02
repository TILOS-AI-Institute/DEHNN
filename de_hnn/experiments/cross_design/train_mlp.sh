#!/bin/bash

program=train_mlp

target=classify
#target=demand
#target=capacity
#target=congestion

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
hidden_dim=60
load_pe=1
# Position encoding
num_eigen=10

# Global information
load_global_info=0

# Persistence diagram & Neighbor list
load_pd=1

# Device
device=cuda
device_idx=1

# Test mode
test_mode=0

for fold in 0
do
name=${program}.target.${target}.num_epoch.${num_epoch}.batch_size.${batch_size}.learning_rate.${learning_rate}.seed.${seed}.hidden_dim.${hidden_dim}.load_pe.${load_pe}.num_eigen.${num_eigen}.load_global_info.${load_global_info}.load_pd.${load_pd}.fold.${fold}
CUDA_VISIBLE_DEVICES=$device_idx python3 $program.py --target=$target --dir=$dir --data_dir=${data_dir} --name=$name --num_epoch=$num_epoch --batch_size=$batch_size --learning_rate=$learning_rate --seed=$seed --hidden_dim=$hidden_dim --load_pe=${load_pe} --num_eigen=${num_eigen} --load_global_info=$load_global_info --load_pd=$load_pd --test_mode=$test_mode --fold=$fold --device=$device
done

