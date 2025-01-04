#!/bin/bash

program=train_transformer_net

target=hpwl
#target=demand
#target=capacity
#target=congestion

mkdir $program
cd $program
mkdir $target
cd ..
dir=./$program/$target/

data_dir=../../data/2023-03-06_data/

num_epoch=1
batch_size=1
learning_rate=0.001
seed=123456789
hidden_dim=64
heads=4
local_heads=1
depth=2

# Global information
load_global_info=1

# Persistence diagram & Neighbor list
load_pd=1

# Device
device=cuda
device_idx=1

# Test mode
test_mode=0

# Position encoding dimension if used
pe_dim=10

# Position encoding
pe_type=lap

for fold in 0 
do
name=${program}.target.${target}.num_epoch.${num_epoch}.batch_size.${batch_size}.learning_rate.${learning_rate}.seed.${seed}.hidden_dim.${hidden_dim}.heads.${heads}.local_heads.${local_heads}.depth.${depth}.pe_type.${pe_type}.pe_dim.${pe_dim}.load_global_info.${load_global_info}.load_pd.${load_pd}.fold.${fold}
CUDA_VISIBLE_DEVICES=$device_idx python3 $program.py --target=$target --dir=$dir --data_dir=${data_dir} --name=$name --num_epoch=$num_epoch --batch_size=$batch_size --learning_rate=$learning_rate --seed=$seed --hidden_dim=$hidden_dim --heads=$heads --local_heads=$local_heads --depth=$depth --test_mode=$test_mode --pe_type=$pe_type --pe_dim=$pe_dim --load_global_info=$load_global_info --load_pd=$load_pd --fold=$fold --device=$device
done

