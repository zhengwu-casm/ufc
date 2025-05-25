import math

import gcnn_model
import data_loading
import tensorflow as tf

import os
import random
import numpy as np
random_seed = 42 
random.seed(random_seed)  # set random seed for python
np.random.seed(random_seed)  # set random seed for numpy
tf.random.set_seed(random_seed)  # set random seed for tensorflow-cpu
# os.environ['TF_DETERMINISTIC_OPS'] = '1'  # set random seed for tensorflow-gpu


# 1. Load Data and split dataset.
train_dataset, val_dataset, test_dataset = data_loading.load_data('D:/PythonProject/function_area_identify_new/github/data/json/samples.json',
                                                                  data_separation=[0.6, 0.2, 0.2])
# Number of samples.
n_train = train_dataset[4]
# Number of classes.
n_class = len(set(train_dataset[2]))
# GeometryDescriptors or Fourier
i_type = 0

# print the Graph structure.
print('Graph structure: {0}x{0} Graph with {1} dimensionaal vecter.'.format(train_dataset[1].shape[1],
                                                                            train_dataset[0].shape[2]))
print('labels set: {}'.format(set(train_dataset[2])))
print('  train dataset:')
print('    vertices   : {0}'.format(train_dataset[0].shape))
print('    adjacencies: {0}'.format(train_dataset[1].shape))
print('    labels     : {0}'.format(train_dataset[2].shape))
print('  validation dataset:')
print('    vertices   : {0}'.format(val_dataset[0].shape))
print('    adjacencies: {0}'.format(val_dataset[1].shape))
print('    labels     : {0}'.format(val_dataset[2].shape))
print('  test dataset:')
print('    vertices   : {0}'.format(test_dataset[0].shape))
print('    adjacencies: {0}'.format(test_dataset[1].shape))
print('    labels     : {0}'.format(test_dataset[2].shape))


# 2. Graph Convolution Parmas Setting.
params = dict()
# The dimensions of vectives.
params['Vs'] = train_dataset[0].shape[2]

# The size of adjacencies (Number of vectices).
params['As'] = train_dataset[1].shape[1]

# The parameters of convolation and pooling layer. Two layers
params['Fs'] = [16, 16, 16, 16, 16]
params['Ks'] = [3, 3, 3, 3, 3]
params['Ps'] = [1, 1, 1, 1, 1]

# The parameters of full connection layer.
params['Cs'] = [16, n_class]

params['filter'] = 'monomial'
# params['filter'] = 'localpool'
params['brelu'] = 'b1relu'
params['pool'] = 'mpool1'#apool1
# params['num_epochs'] = 150
params['num_epochs'] = 50
params['batch_size'] = n_train
params['decay_steps'] = n_train / params['batch_size']
params['eval_frequency'] = n_train

# Hyper-parameters.
params['regularization'] = 5e-4
params['dropout'] = 0.0
params['learning_rate'] = 0.001
params['decay_rate'] = 0.95
params['momentum'] = 0
params['dir_name'] = 'buildings'
# momentum,learning_rate = [0.9, 0.01]
# momentum,learning_rate = [0, 0.002, 0.003, 0.001]
gpu_id = 0
# gpu_id = None
gpus = tf.config.experimental.list_physical_devices('GPU')

if len(gpus) == 0 or gpu_id is None:
    device_id = "/device:CPU:0"
else:
    tf.config.experimental.set_visible_devices(gpus[gpu_id], 'GPU')
    device_id = '/device:GPU:0'

with tf.device(device_id):

    # 3. Train the model and evaluate the model performance
    # 1) The convergence, i.e. the training loss and the classification accuracy on the validation set.
    # 2) The performance, i.e. the classification accuracy on the testing set (to be compared with the training set accuracy to spot overfitting).

    # Define the model collection with different hyperparameters
    #
    # The `gcnn_model_performance` can be used to compactly evaluate multiple models with different hyperparameters.
    gcnn_model_performance = gcnn_model.gcnn_model_performance_testing()

    # Test depth of network structures.
    # for i in [2, 3, 4, 5, 6, 7, 8]:
    pool_max_size = 1
    # for i in [2, 3, 4, 5, 6, 7, 8]:
    for i in [2]:
        layer_count = i
        pool_sizes = []
        k = 0
        while k < i:
            pool_size = math.log2(pool_max_size) - k
            if pool_size <= 1:
                pool_size = 1
            pool_sizes.append(int(pool_size))
            k = k + 1
        # Test feature map size of network structures.
        # for j in [4, 8, 16, 32, 64, 128, 256]:
        for j in [64]:
            # Test the k-order size of network structures.
            # for ks in [1, 2, 3, 4, 5, 6]:
            for ks in [6]:
                print("lays:", i, "cells:", j, "k-order:", ks)
                params['Fs'] = [j] * i
                params['Ks'] = [ks] * i  # 1/2/3/4/5
                # params['Ps'] = [2] * i
                # params['Ps'] = [1, 1, 1, 1, 1]
                params['Ps'] = pool_sizes
                name = 'depth={}cell={}k={}'.format(i, j, ks)
                params['dir_name'] = name
                gcnn_model_performance.test(gcnn_model.gcnn(**params), name, params, train_dataset, val_dataset, test_dataset)


