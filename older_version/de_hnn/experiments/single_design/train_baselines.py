import numpy as np
import math
import random
import pickle
import matplotlib.pyplot as plt
random.seed(123456789)

# For visualization
from utils import *

# Methods
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
import sklearn.gaussian_process as gp

# Metrics
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score

# Mean Absolute Relative Error
def mean_absolute_relative_error(y_truth, y_pred):
    return np.mean(np.abs(y_truth - y_pred) / np.abs(y_truth))

# Learning target
target = 'demand'
# target = 'capacity'
# target = 'congestion'

designs_dict = {
    0 : 'superblue1',
    5 : 'superblue2',
    10: 'superblue3',
    15: 'superblue4',
    21: 'superblue18',
    26: 'superblue19'
}

# Dataset
data_dir = '../../data/2023-03-06_data/'
graph_index = 26

# Analysis
f = open(data_dir + '/' + str(graph_index) + '.net_demand_capacity.pkl', 'rb')
dictionary = pickle.load(f)
f.close()
demand = dictionary['demand']
capacity = dictionary['capacity']
#congestion = demand - capacity

print('Min capacity:', np.min(capacity))
print('Max capacity:', np.max(capacity))
print('Mean capacity:', np.mean(capacity))
print('STD capacity:', np.std(capacity))

# Select the right learning target
if target == 'demand':
    y = demand
elif target == 'capacity':
    y = capacity
elif target == 'congestion':
    y = congestion
else:
    print('Unknown learning target')
    assert False

#y = np.sum(y, axis = 1)
y = np.array(y)
y_min = np.min(y)
y_max = np.max(y)
y_mean = np.mean(y)
y_std = np.std(y)


# Normalization
y = (y - y_mean) / y_std

print('Learning target:', target)
print('Statistics: min =', y_min, ', max =', y_max, ', mean =', y_mean, ', std =', y_std)

# Methods we want to try
method_names = [
    'LR', # Linear Regression
    'Ridge', # Ridge Regression
    #'Linear-SVM', # Linear Support Vector Machine (i.e. linear kernel)
    #'RBF-SVM', # Kernelized SVM with Radial Basis Function kernel
    #'Gaussian-Processes' # Gaussian Processes
]
num_methods = len(method_names)

# Results for Mean Average Error (MAE)
mae_results = [[] for idx in range(num_methods)]

# Results for Root Mean Square Error (RMSE)
rmse_results = [[] for idx in range(num_methods)]

# Results for Mean Absolute Percentage Error (MAPE)
mape_results = [[] for idx in range(num_methods)]

# Results for Mean Absolute Relative Error (MARE)
mare_results = [[] for idx in range(num_methods)]

# Numerical results
results = [[] for idx in range(num_methods)]

# Read features
f = open(data_dir + '/' + str(graph_index) + '.net_features.pkl', 'rb')
dictionary = pickle.load(f)
f.close()
instance_features = dictionary['instance_features']
X = instance_features

print(X.shape)
print(y.shape)

num_samples = X.shape[0]

# Load the split
f = open(str(graph_index) + '.split_net.pkl', 'rb')
dictionary = pickle.load(f)
f.close()

train_indices = dictionary['train_indices']
valid_indices = dictionary['valid_indices']
test_indices = dictionary['test_indices']

num_train = train_indices.shape[0]
num_valid = valid_indices.shape[0]
num_test = test_indices.shape[0]

print('Number of training samples:', num_train)
print('Number of validation samples:', num_valid)
print('Number of testing samples:', num_test)

X_train = X[train_indices, :]
y_train = y[train_indices]

X_valid = X[valid_indices, :]
y_valid = y[valid_indices]

X_test = X[test_indices, :]
y_test = y[test_indices]

# Train and test for each method with this fold
for idx in range(num_methods):
    method_name = method_names[idx]

    # Create the model
    if method_name == 'LR':
        model = LinearRegression()
    elif method_name == 'Ridge':
        # You will need to search for the optimal hyper-parameter
        model = Ridge(alpha = 10.0)
    elif method_name == 'Linear-SVM':
        # You will need to search for the optimal hyper-parameters
        model = SVR(kernel = 'linear', C = 10.0, epsilon = 2)
    elif method_name == 'RBF-SVM':
        # You will need to search for the optimal hyper-parameters
        model = SVR(kernel = 'rbf', C = 10.0)
    elif method_name == 'Gaussian-Processes':
        # You will need to search for the optimal hyper-parameter
        kernel = gp.kernels.ConstantKernel(1.0, (1e-1, 1e3)) * gp.kernels.RBF(1.0, (1e-3, 1e3))
        model = gp.GaussianProcessRegressor(kernel = kernel, alpha = 0.01, normalize_y = True)
    else:
        print('Unsupported method!')
        assert False

    # Fit the model
    model.fit(X_train, y_train)

    # Make prediction
    y_hat = model.predict(X_test)

    # Original scale
    original_y_test = y_test * y_std + y_mean
    original_y_hat = y_hat * y_std + y_mean

    # Evaluate
    mae = mean_absolute_error(y_test, y_hat)
    mse = mean_squared_error(y_test, y_hat)
    rmse = math.sqrt(mse)
    mape = mean_absolute_percentage_error(original_y_test, original_y_hat)
    mare = mean_absolute_relative_error(original_y_test, original_y_hat)

    # Save the result
    mae_results[idx].append(mae)
    rmse_results[idx].append(rmse)
    mape_results[idx].append(mape)
    mare_results[idx].append(mare)

    # Save numerical prediction
    dictionary = {
        'predict': original_y_hat,
        'truth': original_y_test
    }
    results[idx].append(dictionary)

    print('Done', method_name)

# Summary
print('Summary ----------------------------------')
for idx in range(num_methods):
    method_name = method_names[idx]
    print('\n', method_name, ':')

    array = np.array(mae_results[idx]) # Remember to scale back to the original scale
    mae_mean = np.mean(array)
    print('* MAE (normalized targets) =', mae_mean)

    array = np.array(mae_results[idx]) * y_std # Original scale
    mae_mean = np.mean(array)
    print('* MAE =', mae_mean)

    array = np.array(rmse_results[idx]) # Remember to scale back to the original scale
    rmse_mean = np.mean(array)
    print('* RMSE (normalized targets) =', rmse_mean)

    array = np.array(rmse_results[idx]) * y_std # Original scale
    rmse_mean = np.mean(array)
    print('* RMSE =', rmse_mean)

    array = np.array(mape_results[idx]) # Average relative error
    mape_mean = np.mean(array)
    print('* MAPE =', mape_mean)

    array = np.array(mare_results[idx]) # Average relative error
    mare_mean = np.mean(array)
    print('* MARE =', mare_mean)

# Visualization
print('------------------------------------------')
for idx in range(num_methods):
    method_name = method_names[idx]
    design_name = designs_dict[graph_index]
    dictionary = results[idx][0]
    predict = dictionary['predict']
    truth = dictionary['truth']

    output_name = target + '_' + method_name + '_' + design_name + '.png'
    plot_figure(truth, predict, method_name, design_name, output_name)
    print('Created figure', output_name)

print('Done')
