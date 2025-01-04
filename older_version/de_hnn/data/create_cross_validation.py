import numpy as np
import pickle
import random
random.seed(123456789)

abs_dir = '/data/zluo/'
raw_data_dir = abs_dir + 'new_data/'

designs_list = [
    'superblue1',
    'superblue2',
    'superblue3',
    'superblue4',
    'superblue5',
    'superblue6',
    'superblue7',
    'superblue18',
    'superblue19'
]
num_designs = len(designs_list)
num_variants_list = [
    5,
    5,
    6,
    5,
    6,
    6,
    6,
    5,
    6
]
assert num_designs == len(num_variants_list)

# Generate all names
sample_names = []
corresponding_design = []
for idx in range(num_designs):
    for variant in range(num_variants_list[idx]):
        sample_name = raw_data_dir + designs_list[idx] + '/' + str(variant + 1) + '/'
        sample_names.append(sample_name)
        corresponding_design.append(designs_list[idx])

# Folds for cross-validation
N = len(sample_names)
data_dir = '2023-03-06_data/'

num_folds = num_designs
folds = []
for fold in range(num_folds):
    design = designs_list[fold]
    indices = [idx for idx in range(N) if corresponding_design[idx] == design]
    folds.append(indices)

print(folds)

dictionary = {
    'folds': folds
}
fn = data_dir + '/' + str(num_folds) + '_fold_cross_validation.pkl'
f = open(fn, "wb")
pickle.dump(dictionary, f)
f.close()

