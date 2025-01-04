#!/bin/bash

python create_full_data.py
python create_global_information.py
python create_local_pd_data.py
python create_net_demand.py
python create_net_features.py
python create_hpwl.py
python create_degree.py
python create_part_id.py

python create_split.py
python create_split_net.py

