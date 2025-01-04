import numpy as np
import math
import random
import pickle
import sys
from tqdm import tqdm

# Fix random seed
random.seed(123456789)

# Data directory
data_dir = '2023-03-06_data/'

# Index of the graph
graph_index = sys.argv[1]

f = open(data_dir + '/' + str(graph_index) + '.net_hpwl.pkl', 'rb')
dictionary = pickle.load(f)
f.close()
instance_features = np.array(dictionary['hpwl'])

X = instance_features

print('Graph index:', graph_index)
print(X.shape)

num_samples = X.shape[0]
perm = np.arange(num_samples)

from sklearn.model_selection import KFold

kf = KFold(n_splits=4, random_state=123456789, shuffle=True)
idx_num = 1

for train_indices, valid_indices in kf.split(perm):
    print(train_indices.shape, valid_indices.shape)
    

    dictionary = {
        'train_indices': train_indices,
        'valid_indices': valid_indices,
        'test_indices': valid_indices
    }
    f = open('split' + '/' + str(idx_num) + '/' + str(graph_index) + '.split_net.pkl', 'wb')
    pickle.dump(dictionary, f)
    f.close()
    
    idx_num += 1

print('Done')
