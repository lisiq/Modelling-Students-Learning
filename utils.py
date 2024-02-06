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


def create_data_object_heterogeneous(df, return_aux_data=False, item_features=True, student_features=True):
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
    
    if student_features:
        data['student'].x = torch.from_numpy(df_student[['motherTongue',  'Gender']].values).to(torch.float)
        
    rem_dup = df[['code', 'scale']].drop_duplicates()
    rem_dup_index = rem_dup.index
    #data['item'].x = torch.from_numpy(df[['scale']].values)[rem_dup_index].to(torch.float)
    if item_features:
        data['item'].x = scale_features[rem_dup_index] 
        
    df_item = pd.DataFrame({ 'scale' : scales[rem_dup_index], 
                              'matrix': df.matrix[rem_dup_index],
                              'IRT_difficulty': df.IRT_difficulty[rem_dup_index],
                              'matdiff': df.matdiff[rem_dup_index],
                              'topic': df.topic[rem_dup_index],
                              'responseformat': df.responseformat[rem_dup_index],
                              'textlength': df.textlength[rem_dup_index]
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
    
    y_predsoft = pred.squeeze().numpy()#softmax(pred).numpy()[:, 1]
    y_pred = pred.squeeze().round().long().numpy()#.argmax(dim=1, keepdim=True).view(-1).numpy()
    return {
            'AUC':roc_auc_score(y_true, y_predsoft), #
            'Confusion':confusion_matrix(y_true, y_pred).tolist(),
            # 'F1-score-weighted':f1_score(y_true, y_pred, average='weighted'), #
            # 'F1-score-macro':f1_score(y_true, y_pred, average='macro'),
            # 'F1-score-micro':f1_score(y_true, y_pred, average='micro'),
            # 'Accuracy':accuracy_score(y_true, y_pred),
            'Balanced Accuracy': balanced_accuracy_score(y_true, y_pred),
            # 'Precision-weighted':precision_score(y_true, y_pred, average='weighted'), #
            # 'Precision-macro':precision_score(y_true, y_pred, average='macro'),
            # 'Precision-micro':precision_score(y_true, y_pred, average='micro'),
            # 'Recall-weighted':recall_score(y_true, y_pred, average='weighted'), #
            # 'Recall-macro':recall_score(y_true, y_pred, average='macro'),
            # 'Recall-micro':recall_score(y_true, y_pred, average='micro'),
            # 'AMI': adjusted_mutual_info_score(y_true, y_pred)
            }


################ SYNTHETIC

import random


def load_data_synthetic(fname): 
    return torch.load(fname)


def generate_data_object_synthetic(n_students,n_tasks,n_task_per_student,  error_proness_denom = 2, probabilistic=True, number_of_tasks_per_students_is_max=False):

    edge_indices, y, students_ability, code_difficulty= generate_synthetic_student_data_interactions_heterogeneous(n_students,n_tasks,n_task_per_student,  error_proness_denom , probabilistic, number_of_tasks_per_students_is_max)
    
    data = create_data_object_synthetic_heterogeneous(n_students,n_tasks,edge_indices,y, students_ability, code_difficulty)
    return data

def generate_multidimensional_data_object_synthetic(n_students,n_tasks,n_task_per_student,  n_topics = 2 , dimension=2, probabilistic=True, number_of_tasks_per_students_is_max=False):

    edge_indices, y, student_gaussians, item_gaussians, item_difficulty = generate_multidimensional_synthetic_student_data_interactions_heterogeneous(n_students,n_tasks,n_task_per_student,  n_topics, dimension, probabilistic, number_of_tasks_per_students_is_max)
    
    data = create_data_object_synthetic_heterogeneous(n_students,n_tasks,edge_indices,y, student_gaussians, item_gaussians, item_difficulty)
    return data

def polar_to_cartesian(radius, angles):
    n = len(angles)+1
    cartesian_coordinates = np.ones(n) * radius
    for i in range(n-1):
        # print(i)
        for j in range(i):
            # print('sin ',j+1)
            cartesian_coordinates[i] *= np.sin(angles[j])
        # print('cos ',i+1)
        cartesian_coordinates[i] *= np.cos(angles[i])
    # print('sin ',n-1)
    cartesian_coordinates[n-1] *= np.prod([np.sin(angles[i]) for i in range(n-1)])
    return cartesian_coordinates

# Given the raidius and target probability that an item is passed by  a student with competence in that topic, return the corresponding item difficulty
# formula is obtained by reversing the sigmoid function
def get_difficulty_boundaries(radius, target_probability):
    return np.log((1-target_probability)/target_probability) + radius**2


# https://www.psychometrics.cam.ac.uk/system/files/documents/multidimensional-item-response-theory.pdf
def generate_multidimensional_synthetic_student_data_interactions_heterogeneous(
        n_students,
        n_items,
        n_tasks_per_students,
        n_topics=2, 
        dimension = 2, 
        probabilistic=True,
        number_of_tasks_per_students_is_max=False):
    # give random ability and difficulty features to students and tasks respectively
    n_difficulty_levels = 10
    # 2*torch.rand(dataset.x.shape)-1

    radius = np.sqrt(2)
    list_theta = []
    # sampling random angles for the gaussian means
    for i in range(n_topics):
        list_theta.append(np.random.uniform(0, 2*np.pi, dimension-1)) # -1 cause one dimension is fixed by the radius
    # list_theta = [np.array(theta) for theta in [[np.pi/4],[(3/2)*np.pi - np.pi/4]]]
    list_theta = [np.array([2*np.pi*(i/n_topics)]) for i in range(n_topics)]
    gaussian_means = [torch.from_numpy(polar_to_cartesian(radius, theta)).float() for theta in list_theta]
    # print(gaussian_means)

    gaussians = [torch.distributions.multivariate_normal.MultivariateNormal(
        # loc = torch.rand(dimension)*scale,
        loc = gaussian_means[i],
        covariance_matrix = torch.eye(dimension) * 0.1
        ) for i in range(n_topics) ]

    # we sample one vector of multidimensional ability (theta) for each student 
    student_gaussians = [np.random.randint(n_topics) for _ in range(n_students)]
    students_vector = {
        k:gaussians[sg].sample() for k,sg in enumerate(student_gaussians)
    }

    # we sample one vector of multidimensional discrimination for each item 
    item_gaussians = [np.random.randint(n_topics) for _ in range(n_items)]
    item_vector = {
        k: gaussians[ig].sample() for k,ig in enumerate(item_gaussians)
    }
    # we sample a difficult value per item
    difficulty_levels = np.linspace(
        get_difficulty_boundaries(radius, .50), # lower difficulty boundary
        get_difficulty_boundaries(radius, .20), # higher difficulty boundary
        n_difficulty_levels)  # number of difficulty levels

    item_difficulty = {
        # k: random.sample(range(max_difficulty), k=1) for k in range(n_items)
        k: diff for k, diff in enumerate(random.choices(
                    population=list(difficulty_levels), 
                    weights=None, # weights are all equal. However, we could use a different distribution
                    k=n_items))
    }

    edge_indices = []
    y = []
    list_x = []
    for item_id, difficulty in item_difficulty.items():
        # sample random connections with students
        if number_of_tasks_per_students_is_max:
            k = random.sample(list(range(1,n_tasks_per_students)),k=1)[0]
        else:
            k = n_tasks_per_students
        s = random.sample(list(students_vector.keys()), k=k)
        # sample whether item is performed correcly from sigmoid with parameters given by MIRT model
        for student_id in s:
            edge_indices.append((student_id, item_id))
            x = torch.dot(students_vector[student_id],item_vector[item_id]) - difficulty
            if probabilistic:  
                sigmoid_value = 1 / (1 + np.exp(-x))
                list_x.append(sigmoid_value)
                if random.random() <= sigmoid_value:
                    y.append(1)
                else:
                    y.append(0)
            else:
                if x > 0:
                    y.append(1)
                else: 
                    y.append(0)   

    # import matplotlib.pyplot as plt
    # plt.hist(list_x, bins=100)
    # plt.show()
    # bins = torch.bincount(torch.tensor(y))
    # print(bins)
    # print(bins.max()/bins.sum())
    return edge_indices, y, student_gaussians, item_gaussians, item_difficulty 


def generate_synthetic_student_data_interactions_heterogeneous(n_students,n_tasks,n_tasks_per_students, error_proness_denom = 2, probabilistic=True, number_of_tasks_per_students_is_max=False):
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
    for c, difficulty in code.items():
        if number_of_tasks_per_students_is_max:
            k = random.sample(list(range(1,n_tasks_per_students)),k=1)[0]
        else:
            k = n_tasks_per_students
        s = random.sample(list(students.keys()), k=k)
        for i in s:
            edge_indices.append((i, c))
            x = students[i][0] - difficulty[0]
            
            if probabilistic:  
                sigmoid_value = 1 / (1 + np.exp(-x/error_proness_denom))
                if random.random() <= sigmoid_value:
                    y.append(1)
                else:
                    y.append(0)
            else:
                if x > 0:
                    y.append(1)
                else: 
                    y.append(0)   

    
    return edge_indices, y, list(students.values()), list(code.values())


from torch_geometric.data import HeteroData
def create_data_object_synthetic_heterogeneous(n_students,n_tasks,edge_indices,y, student_gaussians, item_gaussians, item_difficulty):
    data  = HeteroData()

    # Save node indices
    data['student'].node_id = torch.arange(n_students)
    data['item'].node_id = torch.arange(n_tasks)

    # Save difficulty and ability
    data['student'].gaussians = torch.tensor(student_gaussians)
    data['item'].gaussians = torch.tensor(item_gaussians)
    data['item'].difficulty = torch.tensor([item_difficulty[i] for i in range(n_tasks)])


    # Add the node features
    # there seems to be students with different mother tongue and gender in different occasions
    data['student'].x= torch.eye(n_students)
    data['item'].x = torch.eye(n_tasks)

    # Add the edge indices
    data['student', 'responds', 'item'].edge_index = torch.from_numpy(np.array(edge_indices).T).to(torch.long)

    #add the edge attrs
    data['student', 'responds', 'item'].edge_attr = torch.tensor([1]*len(y)).to(torch.float).reshape(-1,1)

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
    if hasattr(input_data['student'], 'x'):
        data['student'].x = input_data['student']['x']
    
    if hasattr(input_data['item'], 'x'):
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


