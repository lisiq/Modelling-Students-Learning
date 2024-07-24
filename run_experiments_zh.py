
import os
import itertools
os.environ['OPENBLAS_NUM_THREADS'] = '1'

from manage_experiments import *

import torch
import pandas as pd
import numpy as np
from tqdm import tqdm


parameters = {
    'model_type' : None, #'GNN', # IRT, GNN
    'hidden_dims': None, #[16,8],
    'df_name': None,
    'epochs': 10000,
    'learning_rate': 0.005,
    'weight_decay': None,
    # 'dropout': 0.4,
    'early_stopping': 200,
    'n_splits': 5,
    'device': 'cuda:0',
    'batch_size': 2**16,
    #'neighbours':[-1, -1]
    'neighbours':[50, 50]
    }


df_names = ['mindsteps_set_matrix'] # ['mindsteps_set_full']
hidden_dims_GNN = [[16, 8], [8, 8], [8, 4], [4, 4]]
decoder_dims_GNN = [4, 8, 16, 32]
hidden_dims_IRT = [1, 3, 5]
lambda1s = [0, 0.00001, 0.0001]
lambda2s = [0, 0.00001, 0.0001]
weight_decays = [0]
batch_norms = [False, True]
dropouts = [0, 0.2, 0.4]


repeat_experiment = 1
n_parallel = 1
fold = "results_tuning/"

if __name__ == '__main__':

    # GNN
    for df_name, hidden_dim, decoder_dim, batch_norm, dropout, weight_decay in itertools.product(df_names, hidden_dims_GNN, decoder_dims_GNN, batch_norms, dropouts, weight_decays):
  
        parameters['model_type'] = 'GNN'
        parameters['df_name'] = df_name
        parameters['hidden_dims'] = hidden_dim 
        parameters['decoder_dim'] = decoder_dim 
        parameters['weight_decay'] = weight_decay
        parameters['batch_norm'] = batch_norm
        parameters['dropout'] = dropout
            
        create_tasks(
                parameters,
                repeat_experiment = repeat_experiment,
                folder = fold
        )
        
    # IRT
    for df_name, hidden_dim, lambda1, lambda2 in itertools.product(df_names, hidden_dims_IRT, lambda1s, lambda2s):

        parameters['model_type'] = 'IRT'
        parameters['df_name'] = df_name
        parameters['hidden_dims'] = hidden_dim 
        parameters['lambda1'] = lambda1 
        parameters['lambda2'] = lambda2 
        parameters['weight_decay'] = 0
            
        create_tasks(
                parameters,
                repeat_experiment = repeat_experiment,
                folder = fold
        )
    
    if True:
        list_unfinished = find_unfinished(fold)
        for filename in list_unfinished:
            perform_experiment(filename)
