import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.data import Data, HeteroData
import pandas as pd
import numpy as np
softmax = torch.nn.Softmax(dim=1)


def mymode(x):
    return pd.Series.mode(x, dropna=False)[0]

def load_data_heterogeneous(path):
    """
        loads the data and performs preprocessing steps
    """
    df = pd.read_csv(path + '.csv', index_col=0)
       
    df['matdiff'] = df.matrix.apply(lambda x: x.split('.')[4] if type(x) == 'str' else '')
    df['matcode'] = df.matrix.apply(lambda x: '.'.join(x.split('.')[:4]) if type(x) == 'str' else '')
    df['domain'] = df.scale.apply(lambda x: x[0])
    # code to index starting from 0 since they start from 100+
    code_to_index = {k:v for v,k in zip(range(df.code.nunique()), df.code.unique().tolist())}
    
    # do the mapping 
    df.code = df.code.apply(lambda x: code_to_index[x])
    #df.scale = df.scale.apply(lambda x: scale_to_index[x])    

    print(df.shape)
    return df


def create_data_object_heterogeneous(df, return_aux_data=False):
    data = HeteroData()

    scales = df.scale
    scales_oh = scales.str.get_dummies('|')
    scale_features = torch.from_numpy(scales_oh.values).to(torch.float)

    # Save node indices
    data['student'].node_id = torch.arange(df.studentId.nunique())
    data['item'].node_id = torch.arange(df.code.nunique())

    # Add the node features
    # there seems to be students with different mother tongue and gender in different occasions
    df_student = df.groupby('studentId').agg({'age': 'mean', 'grade': 'mean',
                                              'motherTongue': mymode, 'Gender': mymode}).reset_index()
    #rem_dup = df[['studentId']].drop_duplicates()
    #rem_dup_index_student = rem_dup.index
    #df_student = df[['motherTongue', 'Gender', 'age', 'grade']].iloc[rem_dup_index_student, :]
    data['student'].x = torch.from_numpy(df_student[['motherTongue',  'Gender']].values).to(torch.float)
    rem_dup = df[['code', 'scale']].drop_duplicates()
    rem_dup_index = rem_dup.index
    #data['item'].x = torch.from_numpy(df[['scale']].values)[rem_dup_index].to(torch.float)
    data['item'].x = scale_features[rem_dup_index] 
    df_item = pd.DataFrame({ 'scale' : scales[rem_dup_index], 
                              'matrix': df.matrix[rem_dup_index],
                              'IRT_difficulty': df.IRT_difficulty[rem_dup_index],
                              'matdiff': df.matdiff[rem_dup_index],
                              'topic': df.topic[rem_dup_index],
                              'responseformat': df.responseformat[rem_dup_index],
                              'textlength': df.textlength[rem_dup_index],
                              'ability': df.ability[rem_dup_index]
                           })


    df_item['domain'] = df_item['scale'].apply(lambda x: x[0])


    # Add the edge indices
    data['student', 'responds', 'item'].edge_index = torch.from_numpy(df[['studentId', 'code']].values.T)

    # Add the edge attributes
    df_edge = df[['age', 'grade', 'ability']]
    #df_edge = df[['age', 'grade']].sample(frac=1).reset_index(drop=True)

    data['student', 'responds', 'item'].edge_attr = torch.from_numpy(df[['age', 'grade']].values).to(torch.float)

    # Add the edge label
    #data['student', 'responds', 'item'].edge_label = torch.tensor(df['score'].values)
    data['student', 'responds', 'item'].y = torch.tensor(df['score'].values)

    # We use T.ToUndirected() to add the reverse edges from subject to students 
    # in order to let GNN pass messages in both ways
    # Add a reverse ('item', 'rev_takes', 'student') relation for message passing:
    data = T.ToUndirected()(data)
    del data['item', 'rev_responds', 'student'].edge_attr  
    #del data['item', 'rev_responds', 'student'].edge_label # Remove 'reverse' label.
    del data['item', 'rev_responds', 'student'].y

    if return_aux_data:
        return data, df_student, df_item, df_edge
    else:
        return data



def calculate_metrics(y_true, pred):
    from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, recall_score, precision_score, balanced_accuracy_score, roc_auc_score
    from sklearn.metrics.cluster import adjusted_mutual_info_score
    
    y_predsoft = softmax(pred).numpy()[:, 1]
    y_pred = pred.argmax(dim=1, keepdim=True).view(-1).numpy()
    
    return {
            'AUC':roc_auc_score(y_true, y_predsoft),
            'F1-score-weighted':f1_score(y_true, y_pred, average='weighted'),
            # 'F1-score-macro':f1_score(y_true, y_pred, average='macro'),
            # 'F1-score-micro':f1_score(y_true, y_pred, average='micro'),
            # 'Accuracy':accuracy_score(y_true, y_pred),
            'Balanced Accuracy': balanced_accuracy_score(y_true, y_pred),
            'Precision-weighted':precision_score(y_true, y_pred, average='weighted'),
            # 'Precision-macro':precision_score(y_true, y_pred, average='macro'),
            # 'Precision-micro':precision_score(y_true, y_pred, average='micro'),
            'Recall-weighted':recall_score(y_true, y_pred, average='weighted'),
            # 'Recall-macro':recall_score(y_true, y_pred, average='macro'),
            # 'Recall-micro':recall_score(y_true, y_pred, average='micro'),
            # 'AMI': adjusted_mutual_info_score(y_true, y_pred)
            }


################ SYNTHETIC

import random


def load_data_synthetic(fname): 
    return torch.load(fname)


def generate_data_object_synthetic(n_students,n_tasks,n_student_per_task, probabilistic):

    edge_indices, y= generate_synthetic_student_data_interactions_heterogeneous(n_students,n_tasks,n_student_per_task, probabilistic)
    
    data = create_data_object_synthetic_heterogeneous(n_students,n_tasks,edge_indices,y)
    return data


def generate_synthetic_student_data_interactions_heterogeneous(n_students,n_tasks,n_student_per_task, probabilistic=True):
    # give random ability and difficulty features to students and tasks respectively
    max_difficulty = 10
    students = {
        k: random.sample(range(max_difficulty),k=1) for k in range(n_students)
    }

    code = {
        k: random.sample(range(max_difficulty), k=1) for k in range(n_tasks)
    }

    edge_indices = []
    y = []
    for c, cv in code.items():
        # k = random.sample(list(range(1,n_student_per_task)),k=1)
        k = n_student_per_task
        s = random.sample(students.keys(), k=k)
        for i in s:
            edge_indices.append((i, c))
            x = students[i][0] - cv[0]
            
            if probabilistic:
                sigmoid_value = 1 / (1 + np.exp(-x/2))
                if random.random() <= sigmoid_value:
                    y.append(1)
                else:
                    y.append(0)
            else:
                if x > 0:
                    y.append(1)
                else: 
                    y.append(0)   
    
    return edge_indices, y


from torch_geometric.data import HeteroData
def create_data_object_synthetic_heterogeneous(n_students,n_tasks,edge_indices,y):
    data  = HeteroData()

    # Save node indices
    data['student'].node_id = torch.arange(n_students)
    data['item'].node_id = torch.arange(n_tasks)

    # Add the node features
    # there seems to be students with different mother tongue and gender in different occasions
    data['student'].x= torch.eye(n_students)
    data['item'].x = torch.eye(n_tasks)

    # Add the edge indices
    data['student', 'responds', 'item'].edge_index = torch.from_numpy(np.array(edge_indices).T)

    # Add the edge label
    data['student', 'responds', 'item'].y = torch.from_numpy(np.array(y)).to(torch.long)

    # We use T.ToUndirected() to add the reverse edges from subject to students 
    # in order to let GNN pass messages in both ways
    data = T.ToUndirected()(data)
    del data['item', 'rev_responds', 'student'].edge_attr  # Remove 'reverse' label.
    del data['item', 'rev_responds', 'student'].y  # Remove 'reverse' label.
    return data


# create subgraph
def subgraph(input_data, index):
    data  = HeteroData()

    # Save node indices
    data['student'].node_id = input_data['student']['node_id']
    data['item'].node_id = input_data['item']['node_id']

    # Add the node features
    # there seems to be students with different mother tongue and gender in different occasions
    data['student'].x= input_data['student']['x']
    data['item'].x = input_data['item']['x']

    # Add the edge indices
    data['student', 'responds', 'item'].edge_index = input_data['student', 'responds', 'item'].edge_index[:, index]

        
    try: 
        # Add the edge attrs
        data['student', 'responds', 'item'].edge_attr = input_data['student', 'responds', 'item'].edge_attr[index]
    except AttributeError:
        pass

    # Add the edge label
    data['student', 'responds', 'item'].y =  input_data['student', 'responds', 'item'].y[index]

    # We use T.ToUndirected() to add the reverse edges from subject to students 
    # in order to let GNN pass messages in both ways
    data = T.ToUndirected()(data)

    try:
        del data['item', 'rev_responds', 'student'].edge_attr  # Remove 'reverse' label.
    except AttributeError:
        pass    
    del data['item', 'rev_responds', 'student'].y  # Remove 'reverse' label.
    return data


################ DEPRECATED



def load_data_heterogeneous_deprecated(path):
    """
        loads the data and performns preprocessing steps
    """
    df = pd.read_csv(path+'.csv', index_col=0)

    # code to index starting from 0 since they start from 100+
    code_to_index = {k:v for v,k in zip(range(df.code.nunique()), df.code.unique().tolist())}
    # map scale to index
    scale_to_index = {k:v for v,k in zip(range(df.scale.nunique()), df.scale.unique().tolist())}
    # do the mapping 
    df.code = df.code.apply(lambda x: code_to_index[x])
    df.scale = df.scale.apply(lambda x: scale_to_index[x])   
    
    # map scale from ordinal to one-hot-encoding
    df = pd.concat([df, pd.get_dummies(df.scale, prefix='scale')], axis=1) 
    return df

def create_data_object_heterogeneous_deprecated(df):
    data  = HeteroData()
    # Save node indices
    data['student'].node_id = torch.arange(df.studentId.nunique())
    data['code'].node_id = torch.arange(df.code.nunique())

    # Add the node features
    rem_dup = df[['studentId']].drop_duplicates()
    rem_dup_index = rem_dup.index
    data["student"].x = torch.from_numpy(df[['motherTongue', 'Gender']].values)[rem_dup_index].to(torch.float)
    rem_dup = df[['code', 'scale']].drop_duplicates()
    rem_dup_index = rem_dup.index
    data["code"].x = torch.from_numpy(df[['scale_0', 
                                          'scale_1', 
                                          'scale_2', 
                                          'scale_3', 
                                          'scale_4', 
                                          'scale_5',
                                          'scale_6', 
                                          'scale_7', 
                                          'scale_8', 
                                          'scale_9', 
                                          'scale_10']].values)[rem_dup_index].to(torch.float)

    # Add the edge indices
    data['student', "takes", "code"].edge_index = torch.from_numpy(df[['studentId', 'code']].values.T)
    # Add the edge attributes
    data['student', "takes", "code"].edge_attr = torch.from_numpy(df[['age', 'grade']].values).to(torch.float)
    # Add the edge label
    data['student', "takes", "code"].y = torch.from_numpy(df.score.values.T).to(torch.long)

    # We use T.ToUndirected() to add the reverse edges from subject to students 
    # in order to let GNN pass messages in both ways
    # Add a reverse ('code', 'rev_takes', 'student') relation for message passing:
    data = T.ToUndirected()(data)
    del data['code', 'rev_takes', 'student'].edge_attr  # Remove "reverse" label.
    del data['code', 'rev_takes', 'student'].y
    return data

