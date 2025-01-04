import pickle
import numpy as np
from tqdm import tqdm

data_dir = '2023-03-06_data' 

for sample in tqdm(range(32)):
    with open(f"{data_dir}/{sample}.node_features.pkl", "rb") as f:
        d = pickle.load(f)

    pos_lst = d['instance_features'][:, :2]
    #x_lst = pos_lst[:, 0].T
    #y_lst = pos_lst[:, 1].T
    #unit_width = abs(max(x_lst) - min(x_lst))/10
    #unit_height = abs(max(y_lst) - min(y_lst))/10

    part_dict = {}
    phy_id_set = set()

    for idx in range(len(pos_lst)):
        pos = pos_lst[idx]
        x = int(pos[0]//0.1)
        y = int(pos[1]//0.1)
        part_id = x * 10 + y
        part_dict[idx] = part_id
        phy_id_set.add(part_id)
    

    part_to_idx = {val:idx for idx, val in enumerate(phy_id_set)}
    part_dict = {idx:part_to_idx[part_id] for idx, part_id in part_dict.items()}
    file_name = data_dir + '/' + str(sample) + '.star_part_dict.pkl'
    f = open(file_name, 'wb')
    dictionary = pickle.dump(part_dict, f)
    f.close()
