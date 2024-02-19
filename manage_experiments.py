from Heterogeneous_temporal_embedder import EmbedderHeterogeneous, train_embedder_heterogeneous, test_embedder_heterogeneous
from sklearn.model_selection import KFold, train_test_split
from tqdm import tqdm
import numpy as np
import torch
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import subgraph
from utils import *
import json
import os
import numpy as np
import uuid as _uuid
import glob as _glob
from IRT import MIRT_2PL, train_IRT, test_IRT
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt

import random 
import sys

random.seed(0)

random_state = random.randint(0, 2**32 - 1)

def save_dict(data, filename):
	"""
		saves dictionary data
	"""
	with open(filename, "w") as file:
		file.write(json.dumps(data))

def create_tasks( parameters, repeat_experiment, folder = r"res/",):
	"""
		creates files that are treated as tasks to compute
		folder has to be in form r"res/"
		
	"""
	parameters["done"] = False
	for _ in range(repeat_experiment):
		# write file
		filename_input =  folder + str(_uuid.uuid4()) + ".results"
		save_dict(parameters, filename_input)


def find_unfinished(folder):
	"""
		finds all tasks that are not finished.
		The tasks are assumed to be created with
		create_tasks.
		folder has form r"res/"
	"""
	unfinished = []
	for filename in _glob.glob(folder+"*.results"):
		with open(filename, "r") as file:
			results = json.load(file)
			if not results["done"]:
				unfinished.append(filename)
	return unfinished


def perform_experiment(filename):
    with open(filename, "r") as file:
        parameters = json.load(file)

    if parameters["df_name"] in ["synthetic.salamoia"]:
         data = load_data_synthetic(parameters["df_name"])
    elif "mindsteps" in parameters["df_name"]:
        df = load_data_heterogeneous("data/" + parameters["df_name"])
        data = create_data_object_heterogeneous(df)
    else:
          assert False, f"unknown dataset: {parameters['df_name']}"
        
    output_dict = perform_cross_validation(data, parameters)
        
    output_dict['done'] = True
    # print the keys and get the type of all thevalues in the dictionary
    for key, value in output_dict.items():
        print(key, type(value))
    
    save_dict(output_dict, filename)    


####################################################

# function to get the embedings
# using class methods instead
#def get_embedding(model, data):
#    model.eval()

#    x_dict = {
#          'student': model.student_lin(data['student'].x) +  model.student_emb(data['student'].node_id),
#          'item': model.item_lin(data['item'].x) + model.item_emb(data['item'].node_id),
#        } 
    
#    new_x = torch.cat([x_dict['student'], x_dict['item']], dim=0)
#    return new_x.detach().cpu().numpy().tolist()


def perform_cross_validation(data, parameters, save_embeddings=False, save_subgraph=False, model=None, final_fit=False):
    n_splits = parameters['n_splits']
    device = torch.device(parameters['device'] if torch.cuda.is_available() else 'cpu')
    kf = KFold(n_splits=n_splits, random_state=random_state, shuffle=True)
    print('WARNING: running with a fixed random state')
    output_dict = {}

    for fold, (train_index, test_index) in tqdm(enumerate(kf.split(range(data['student', 'responds', 'item'].y.size(0))))):

        if final_fit and fold == 1: # for final fit, just fit once
            break

        _, _, test_index, val_index = train_test_split(test_index,
                                        test_index,
                                        test_size=0.5, 
                                        random_state=random_state)
        
        # define train test val subgraphs
        train_subgraph_data = subgraph(data, train_index)
        test_subgraph_data = subgraph(data, test_index).to(device)
        val_subgraph_data = subgraph(data, val_index).to(device)                                        
        
        if 'neighbours' not in parameters.keys():
            neighbours = [10, 10]
        else:
            neighbours = parameters['neighbours']
        train_loader = NeighborLoader(train_subgraph_data, 
                                    num_neighbors = {key: neighbours for key in train_subgraph_data.edge_types}, 
                                    input_nodes=('student', train_subgraph_data['student'].node_id),
                                    directed=True,
                                    replace=False,
                                    batch_size=parameters['batch_size'])
        
        # Initialise
        # this makes the model initalise only once,
        # all the other folds will use only the model defined in the first fold
        # we removed this if statement as we were not sure where you need this
        # if model is None:
            
        if parameters['model_type'] == 'GNN':
            test_loop = test_embedder_heterogeneous
            train_loop = train_embedder_heterogeneous
            
            student_inchannel = data['student'].x.size(1) if hasattr(data['student'], 'x') else None
            item_inchannel = data['item'].x.size(1) if hasattr(data['item'], 'x') else None
            
            if parameters['df_name'] in ['synthetic.salamoia']:
                    model = EmbedderHeterogeneous( 
                    n_students =  data['student'].node_id.size(0),
                    n_items = data['item'].node_id.size(0),
                    student_inchannel = student_inchannel,
                    item_inchannel = item_inchannel,
                    hidden_channels = parameters['hidden_dims'],
                    edge_channel = None,
                    metadata=data.metadata()
                    ).to(device)
            else:
                edge_dim = data['student', 'responds', 'item'].edge_attr.shape[1]
                model = EmbedderHeterogeneous( 
                    n_students = data['student'].node_id.size(0) if not 'student_id' in data else len(np.unique(data['student'].x[:, -1])),
                    n_items = data['item'].node_id.size(0),
                    student_inchannel = student_inchannel,
                    item_inchannel = item_inchannel,
                    hidden_channels = parameters['hidden_dims'],
                    edge_channel = edge_dim,
                    metadata = data.metadata()
                    ).to(device)
                
        elif parameters['model_type'] == 'IRT':
            edge_dim = data['student', 'responds', 'item'].edge_attr.shape[1]
            lambda1 = parameters['lambda1'] if 'lambda1' in parameters else 0
            lambda2 = parameters['lambda2'] if 'lambda2' in parameters else 0
            
            model = MIRT_2PL(parameters['hidden_dims'], edge_dim, data, 
                                lambda1 = lambda1,
                                lambda2 = lambda2).to(device)
            test_loop = test_IRT
            train_loop = train_IRT
                

        optimizer = torch.optim.Adam(model.parameters(), lr=parameters['learning_rate'], weight_decay=parameters['weight_decay'])
        cw = train_subgraph_data['student', 'item'].y.numpy()
        # class_weights = compute_class_weight('balanced', classes=np.unique(cw), y=cw)
        # class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
        pos = cw.sum()
        total_samples = len(cw)
        # class_counts = [total_samples - pos, pos]  
        # class_weights = [total_samples / (2 * count) for count in class_counts]
        # class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

        # Training the model
        losses = []
        best_val_acc = final_test_acc = 0
        early_stopping = 0
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([(total_samples - pos)/pos]).to(device))#class_weights)
        for epoch in tqdm(range(1, parameters['epochs']+1)):
            for batch in tqdm(train_loader, mininterval=30): 
                batch = batch.to(device)
                loss = train_loop(
                    model,
                    batch, 
                    optimizer,
                    criterion,
                    #class_weights
                    )
            losses.append(loss.detach().item())
        
            val_b = test_loop(model, val_subgraph_data, fold, 'val')
            test_b = test_loop(model, test_subgraph_data, fold, 'test')

            if val_b['Balanced Accuracy'+f'_{fold}_val'] > best_val_acc:
                early_stopping = 0

                best_val_acc = val_b['Balanced Accuracy'+f'_{fold}_val']
                final_test_acc = test_b['Balanced Accuracy'+f'_{fold}_test']

                print(f'\nEpoch: {epoch:03d}, Loss: {loss:.4f}, '
                    f'Val: {best_val_acc:.4f}, Test: {final_test_acc:.4f}')

                val_b_ = test_loop(model, val_subgraph_data,  fold, 'val')
                test_b_ = test_loop(model, test_subgraph_data, fold, 'test')

                if save_embeddings: 
                    saved_embedding = model.get_embeddings(data.to(device))      

            else:
                early_stopping += 1

            if early_stopping == parameters['early_stopping']:                
                break
            
            data = data.to('cpu')

        # get the dictionary of model parameters
        # print(plt.imshow(model.encoder.layers[0].state_dict()['student__responds__item.lin_r.weight'].cpu().detach().numpy()))
        # print(model.encoder.layers[0].parameters())
        # assert False
        # print(model.encoder.layers[0].weight)
        

        # Results
        losses_dict = {f'losses_{fold}': losses}
        # Comment this out to save also the embeddings
        best_train_acc = test_loop(model, train_subgraph_data.to(device), fold, 'train')
        print(f'Train balanced accuracy:{best_train_acc["Balanced Accuracy"+f"_{fold}_train"]:.4f}')
        output_dict.update({**parameters,
                            **val_b_, 
                            **test_b_,
                            **best_train_acc,  
                            **losses_dict
                            }) 
        
        if save_embeddings:
            embedding = {f'embedding_{fold}': saved_embedding}
            indices_dict = {f'indices_{fold}': (train_index, val_index, test_index)}            
            output_dict.update({**embedding, **indices_dict})
            
        if save_subgraph:
            output_dict['train_subgraph_data'] = train_subgraph_data
            output_dict['val_subgraph_data'] = val_subgraph_data
            output_dict['test_subgraph_data'] = test_subgraph_data
            
        output_dict['device'] = str(output_dict['device'])
        
    if not final_fit:
        return output_dict
    else: 
        return output_dict, model
