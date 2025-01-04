#!/bin/bash

for fold in {0..73}; 
do
python3 create_split_net.py $fold
done
