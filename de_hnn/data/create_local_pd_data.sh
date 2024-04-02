#!/bin/bash
# 3 8 13 24 
# 30 36 42 47
# 52 58 64 70
for fold in 13 24;
do
    python3 create_local_pd_data.py $fold
done
