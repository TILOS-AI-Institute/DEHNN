#!/bin/bash
program=train_netlistgnn_net

target=$2
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

# NetlistGNN
n_layer=4
node_feats=64 #64
net_feats=64 #128
pin_feats=8
edge_feats=2

# Device
device=cuda
device_idx=$3

# Position encoding
pe=lap
pos_dim=10

# Global information
load_global_info=0

# Persistence diagram & Neighbor list
load_pd=1

# Test mode
test_mode=0

for fold in 0 
do
name=${program}.${target}.n_layer.${n_layer}.node_feats.${node_feats}.net_feats.${net_feats}.pin_feats.${pin_feats}.edge_feats.${edge_feats}.num_epoch.${num_epoch}.batch_size.${batch_size}.learning_rate.${learning_rate}.seed.${seed}.pe.${pe}.pos_dim.${pos_dim}.load_global_info.${load_global_info}.load_pd.${load_pd}.fold.${fold}
CUDA_VISIBLE_DEVICES=$device_idx python3 $program.py --target=$target --n_layer=$n_layer --node_feats=$node_feats --net_feats=$net_feats --pin_feats=$pin_feats --edge_feats=$edge_feats --dir=$dir --name=$name --num_epoch=$num_epoch --batch_size=$batch_size --learning_rate=$learning_rate --seed=$seed --pe=$pe --pos_dim=$pos_dim --load_global_info=$load_global_info --load_pd=$load_pd --test_mode=$test_mode --device=$device --data_dir=$data_dir --fold=$fold
done
