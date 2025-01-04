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
#target = 'congestion'

# Dataset
data_dir = '../../data/2023-03-06_data/'
num_samples = 32

designs_list = [
    'superblue1',
    'superblue2',
    'superblue3',
    'superblue4',
    'superblue18',
    'superblue19'
]

# Analysis
y = []
for sample in range(num_samples):
    f = open(data_dir + '/' + str(sample) + '.net_demand_capacity.pkl', 'rb')
    dictionary = pickle.load(f)
    f.close()
    demand = dictionary['demand']
    capacity = dictionary['capacity']
    #congestion = demand - capacity

    # Select the right learning target
    if target == 'demand':
        y.append(demand)
    elif target == 'capacity':
        y.append(capacity)
    elif target == 'congestion':
        y.append(congestion)
    else:
        print('Unknown learning target')
        assert False

y = np.concatenate(y, axis = 0)
y_min = np.min(y)
y_max = np.max(y)
y_mean = np.mean(y)
y_std = np.std(y)

print('Learning target:', target)
print('Statistics: min =', y_min, ', max =', y_max, ', mean =', y_mean, ', std =', y_std)

# Folds information
f = open(data_dir + '/6_fold_cross_validation.pkl', 'rb')
dictionary = pickle.load(f)
f.close()

folds = dictionary['folds']
num_folds = len(folds)

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

# For each fold
for fold in range(num_folds):
    print('Fold', fold, '-----------------------------')
    test_indices = folds[fold]
    train_indices = [idx for idx in range(num_samples) if idx not in test_indices]
    print('Test indices:', test_indices)
    print('Train indices:', train_indices)

    X_test = []
    y_test = []
    
    for index in test_indices:
        f = open(data_dir + '/' + str(index) + '.net_features.pkl', 'rb')
        dictionary = pickle.load(f)
        f.close()
        instance_features = dictionary['instance_features']
        X_test.append(instance_features)

        f = open(data_dir + '/' + str(index) + '.net_demand_capacity.pkl', 'rb')
        dictionary = pickle.load(f)
        f.close()
        demand = dictionary['demand']
        capacity = dictionary['capacity']
        #congestion = demand - capacity

        if target == 'demand':
            y_test.append(demand)
        elif target == 'capacity':
            y_test.append(capacity)
        elif target == 'congestion':
            y_test.append(congestion)
        else:
            print('Unknown learning target')
            assert False
    
    print('Done reading test set')

    X_train = []
    y_train = []

    for index in train_indices:
        f = open(data_dir + '/' + str(index) + '.net_features.pkl', 'rb')
        dictionary = pickle.load(f)
        f.close()
        instance_features = dictionary['instance_features']
        X_train.append(instance_features)

        f = open(data_dir + '/' + str(index) + '.net_demand_capacity.pkl', 'rb')
        dictionary = pickle.load(f)
        f.close()
        demand = dictionary['demand']
        capacity = dictionary['capacity']
        #congestion = demand - capacity

        if target == 'demand':
            y_train.append(demand)
        elif target == 'capacity':
            y_train.append(capacity)
        elif target == 'congestion':
            y_train.append(congestion)
        else:
            print('Unknown learning target')
            assert False

    print('Done reading train set')

    X_test = np.concatenate(X_test, axis = 0)
    y_test = np.concatenate(y_test, axis = 0)
    X_train = np.concatenate(X_train, axis = 0)
    y_train = np.concatenate(y_train, axis = 0)
    # Normalization
    y_train = (y_train - y_mean) / y_std
    y_test = (y_test - y_mean) / y_std

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
    mae_std = np.std(array)
    print('* MAE (normalized targets) =', mae_mean, '+/-', mae_std)

    array = np.array(mae_results[idx]) * y_std # Original scale
    mae_mean = np.mean(array)
    mae_std = np.std(array)
    print('* MAE =', mae_mean, '+/-', mae_std)

    array = np.array(rmse_results[idx]) # Remember to scale back to the original scale
    rmse_mean = np.mean(array)
    rmse_std = np.std(array)
    print('* RMSE (normalized targets) =', rmse_mean, '+/-', rmse_std)

    array = np.array(rmse_results[idx]) * y_std # Original scale
    rmse_mean = np.mean(array)
    rmse_std = np.std(array)
    print('* RMSE =', rmse_mean, '+/-', rmse_std)

    array = np.array(mape_results[idx]) # Average relative error
    mape_mean = np.mean(array)
    mape_std = np.std(array)
    print('* MAPE =', mape_mean, '+/-', mape_std)

    array = np.array(mare_results[idx]) # Average relative error
    mare_mean = np.mean(array)
    mare_std = np.std(array)
    print('* MARE =', mare_mean, '+/-', mare_std)

# Visualization
print('------------------------------------------')
for idx in range(num_methods):
    method_name = method_names[idx]
    assert len(results[idx]) == len(designs_list)
    
    for fold in range(len(designs_list)):
        design_name = designs_list[fold]
        dictionary = results[idx][fold]
        predict = dictionary['predict']
        truth = dictionary['truth']

        output_name = target + '_' + method_name + '_' + design_name + '.png'
        plot_figure(truth, predict, method_name, design_name, output_name)
        print('Created figure', output_name)

print('Done')
