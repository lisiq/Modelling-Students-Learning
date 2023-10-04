from Heterogeneous_embedder import EmbedderHeterogeneous, train_embedder_heterogeneous, test_embedder_heterogeneous
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
    elif parameters["df_name"] in ["mindsteps_set_full", "mindsteps_subset"]:
        df = load_data_heterogeneous(parameters["df_name"])
        data = create_data_object_heterogeneous(df)
    else:
          assert False, f"unknown dataset: {parameters['df_name']}"
    
    
    output_dict = perform_cross_validation(data, parameters)
    
    
    output_dict["done"] = True
    save_dict(output_dict, filename)    


####################################################

# Benji we have this function to get the embedings
# feel free to use it or to subsitute it to your needs :)
def get_embedding(model, data):
    model.eval()
    x_dict = {
            "student": model.student_lin(data.student_x) +  model.student_emb(data.student_node_id),
            "code": model.code_lin(data.code_x) + model.code_emb(data.code_node_id),
            } 

    new_x = torch.cat([x_dict['student'], x_dict['code']], dim=0)
    return new_x.detach().cpu().numpy().tolist()


def perform_cross_validation(data, parameters):
    n_splits = parameters["n_splits"]
    device = parameters["device"]
    kf = KFold(n_splits=n_splits, random_state=random_state, shuffle=True)
    print("WARNING: running with a fixed random state")
    output_dict = {}

    for fold, (train_index, test_index) in tqdm(enumerate(kf.split(range(data['student', "takes", "code"].y.size(0))))):

        _, _, test_index, val_index = train_test_split(test_index,
                                        test_index,
                                        test_size=0.5, 
                                        random_state=random_state)
        

        # define train test val subgraphs
        train_subgaph_data = subgraph(data, train_index)
        test_subgaph_data = subgraph(data, test_index).to(device)
        val_subgaph_data = subgraph(data, val_index).to(device)
                                        
        
        train_loader = NeighborLoader(train_subgaph_data, 
                                    num_neighbors = {key: [-1] for key in train_subgaph_data.edge_types}, 
                                    input_nodes=('student', train_subgaph_data['student'].node_id),
                                    directed=False,
                                    replace=False,
                                    batch_size=parameters['batch_size'])
        
        
        # Initialise
        if parameters['df_name'] in ['synthetic.salamoia']:
             model = EmbedderHeterogeneous( 
                n_students =  data["student"].x.size(0),
                n_items = data["code"].x.size(0),
                student_inchannel = data["student"].x.size(1),
                item_inchannel = data["code"].x.size(1),
                hidden_channels=parameters['hidden_dims'],
                edge_channel=None,
                metadata=data.metadata()
                ).to(device)
        else:
            model = EmbedderHeterogeneous( 
                n_students =  data["student"].x.size(0),
                n_items = data["code"].x.size(0),
                student_inchannel = data["student"].x.size(1),
                item_inchannel = data["code"].x.size(1),
                hidden_channels=parameters['hidden_dims'],
                edge_channel=data['student', 'takes', 'code'].edge_attr.shape[1],
                metadata=data.metadata()
                ).to(device)


        optimizer = torch.optim.Adam(model.parameters(), lr=parameters['learning_rate'], weight_decay=parameters['weight_decay'])

        # Training the model
        losses = []
        best_val_acc = final_test_acc = 0
        early_stopping = 0

        for epoch in tqdm(range(1, parameters['epochs']+1)):
            for batch in tqdm(train_loader): 
                batch = batch.to(device)
                loss = train_embedder_heterogeneous(
                    model,
                    batch, 
                    optimizer,
                    )
            losses.append(loss.detach().item())
        
            val_b = test_embedder_heterogeneous(model, test_subgaph_data, fold, 'val')
            test_b = test_embedder_heterogeneous(model, val_subgaph_data, fold, 'test')

            if val_b['Balanced Accuracy'+f"_{fold}_val"] > best_val_acc:
                early_stopping = 0

                best_val_acc = val_b['Balanced Accuracy'+f"_{fold}_val"]
                final_test_acc = test_b['Balanced Accuracy'+f"_{fold}_test"]

                print(f'\nEpoch: {epoch:03d}, Loss: {loss:.4f}, '
                    f'Val: {best_val_acc:.4f}, Test: {final_test_acc:.4f}')

                val_b_ = test_embedder_heterogeneous(model, val_subgaph_data,  fold, 'val')
                test_b_ = test_embedder_heterogeneous(model, test_subgaph_data, fold, 'test')

                # Benji you can comment this out to save aslo the embeddings
                # saved_embedding = get_embedding(model, data)
            

            else:
                early_stopping += 1

            if early_stopping == parameters['early_stopping']:
                break
            
            data = data.to("cpu")

        # Results
        losses_dict = {f'losses_{fold}': losses}
        # Benji you can comment this out to save aslo the embeddings
        # embedding = {f'embedding_{fold}': saved_embedding}
        output_dict.update({**parameters,
                            **val_b_, 
                            **test_b_,  
                            **losses_dict}) 
        
        # Benji you can comment this out to save aslo the embeddings
        #, **embedding})
        output_dict["device"] = str(output_dict["device"])
    return output_dict

