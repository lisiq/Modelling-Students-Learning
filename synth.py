import torch
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
from manage_experiments import *

from utils import generate_data_object_synthetic

parameters = {
    "hidden_dims": [16, 8],
    'model_type': "GNN",
    "df_name": "synthetic.salamoia",
    "method": "EdgeClassifier",
    "epochs": 10000,
    "learning_rate": 0.005,
    "weight_decay": 0,
    "dropout": 0.4,
    "early_stopping": 200,
    "n_splits": 3,
    "device": "cuda",
    "done": False,
    "batch_size":128,
    'n_students' :720, # 72% of nodes were students
    'n_tasks':280,
    'max_n_tasks_per_student':280, 
    'min_n_tasks_per_student':5,
    'step_n_tasks_per_student':10,
    'error_proness_denom':3,
    'probabilistic':1,
    'number_of_tasks_per_students_is_max':0
    #
    }

data = generate_data_object_synthetic(
        n_students = parameters['n_students'], 
        n_tasks = parameters['n_tasks'],
        n_task_per_student = 10,
        error_proness_denom = parameters['error_proness_denom'],
        probabilistic = parameters['probabilistic'],
        number_of_tasks_per_students_is_max=parameters['number_of_tasks_per_students_is_max'])


if __name__ == '__main__':
    perform_cross_validation(data, parameters)