
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

from manage_experiments import *

import torch
import pandas as pd
import numpy as np
from tqdm import tqdm


parameters = {
    'model_type' : 'GNN', # IRT, GNN
    'hidden_dims': [16,8],
    'df_name': None,
    'epochs': 10000,
    'learning_rate': 0.005,
    'weight_decay': 0,
    # 'dropout': 0.4,
    'early_stopping': 200,
    'n_splits': 10,
    'device': 'cuda',
    'batch_size': 1024
    }


df_names = ['mindsteps_set_full']

repeat_experiment = 1
n_parallel = 1
fold = "results/"

if __name__ == '__main__':
    for df_name in df_names:
        parameters["df_name"] = df_name
        create_tasks(
                parameters,
                repeat_experiment = repeat_experiment,
                folder = fold
        )

    list_unfinished = find_unfinished(fold)
    for filename in list_unfinished:
        perform_experiment(filename)
