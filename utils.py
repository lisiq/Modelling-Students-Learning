import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.data import Data, HeteroData
import pandas as pd
import numpy as np
softmax = torch.nn.Softmax(dim=1)


def mymode(x):
    return pd.Series.mode(x, dropna=False)[0]

def load_data_heterogeneous(path, y_vars=['score'], domains=None):
    """
        loads the data and performs preprocessing steps
    """
    df = pd.read_csv(path + '.csv', index_col=0)
    df = df.dropna(subset=y_vars)

        
    if 'viewingTime' in y_vars:
        df = df.loc[df.viewingTime > 0.5]
        df = df.loc[df.viewingTime < 500]
        
    df['matdiff'] = df.matrix.apply(lambda x: x.split('.')[4] if type(x) == 'str' else '')
    df['matcode'] = df.matrix.apply(lambda x: '.'.join(x.split('.')[:4]) if type(x) == 'str' else '')
    df['domain'] = df.scale.apply(lambda x: x[0])
    if domains is not None:
        df = df.loc[df.domain.isin(domains)]        

    # code to index starting from 0 since they start from 100+
    code_to_index = {k:v for v,k in zip(range(df.code.nunique()), df.code.unique().tolist())}
    student_to_index = {k:v for v,k in zip(range(df.studentId.nunique()), df.studentId.unique().tolist())}
    
    # do the mapping 
    df.code = df.code.apply(lambda x: code_to_index[x])
    df.studentId = df.studentId.apply(lambda x: student_to_index[x])
    #df.scale = df.scale.apply(lambda x: scale_to_index[x])    
    df = df.reset_index()
    return df


def create_data_object_heterogeneous(df, return_aux_data=False, item_features=True, student_features=True,
                                    undirected=True, y_vars=['score'], forll=False):
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
                                              'motherTongue': mymode, 'Gender': mymode,
                                              'viewingTime':'min'}).reset_index()
    
    if student_features:
        if 'viewingTime' in y_vars:
            data['student'].x = torch.from_numpy(df_student[['viewingTime', 'motherTongue',  'Gender']].values).to(torch.float)
        else:
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
    from torch_geometric.utils import contains_self_loops
    print(contains_self_loops(data['student', 'responds', 'item'].edge_index)) 
    # Add the edge attributes
    df_edge = df[['age', 'grade', 'ability']]
    #df_edge = df[['age', 'grade']].sample(frac=1).reset_index(drop=True)

    if forll:
        data['student', 'responds', 'item'].edge_attr = torch.from_numpy(df[['age', 'grade'] + y_vars].values).to(torch.float)
    else:
        data['student', 'responds', 'item'].edge_attr = torch.from_numpy(df[['age', 'grade']].values).to(torch.float)
        
        # Add the edge label
        #data['student', 'responds', 'item'].edge_label = torch.tensor(df['score'].values)
    data['student', 'responds', 'item'].y = torch.tensor(df[y_vars].values).squeeze()

    if undirected:
        # We use T.ToUndirected() to add the reverse edges from subject to students 
        # in order to let GNN pass messages in both ways
        # Add a reverse ('item', 'rev_takes', 'student') relation for message passing:
        data = T.ToUndirected()(data)
        del data['item', 'rev_responds', 'student'].edge_attr  
        #del data['item', 'rev_responds', 'student'].edge_label # Remove 'reverse' label.
        #if not forll:
        #    del data['item', 'rev_responds', 'student'].y

    if return_aux_data:
        return data, df_student, df_item, df_edge
    else:
        return data


def get_roc_auc_score(y_true, y_predsoft):
    
    from sklearn.metrics import roc_auc_score
    try:
        r = roc_auc_score(y_true, y_predsoft)
    except:
        r = np.nan   
        
    return r


def calculate_metrics(y_true, pred, bernoulli=False, NREPS=100):
    from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, recall_score, precision_score, balanced_accuracy_score
    from sklearn.metrics.cluster import adjusted_mutual_info_score
    #print(y_true.shape)
    #print(pred.size())
    y_predsoft = pred.squeeze().numpy()#softmax(pred).numpy()[:, 1]
    y_pred = pred.squeeze().round().long().numpy()#.argmax(dim=1, keepdim=True).view(-1).numpy()
    
    ba = balanced_accuracy_score(y_true, y_pred)
    pred[pred<0.] = 0.
    pred[pred>1.] = 1.
    ba_ber = np.mean([ balanced_accuracy_score(y_true, torch.bernoulli(pred).squeeze().long().numpy()) for x in range(NREPS)])
    
    if bernoulli:
        ba = ba_ber

    #print('***********')
    #print(y_true[:40])
    #print(y_pred[:40])
    #print('***********')
    return {
            'AUC':get_roc_auc_score(y_true, y_predsoft), #
            'Confusion':confusion_matrix(y_true, y_pred).tolist(),
            # 'F1-score-weighted':f1_score(y_true, y_pred, average='weighted'), #
            # 'F1-score-macro':f1_score(y_true, y_pred, average='macro'),
            # 'F1-score-micro':f1_score(y_true, y_pred, average='micro'),
            # 'Accuracy':accuracy_score(y_true, y_pred),
            'Balanced Accuracy': ba,
            'Balanced Accuracy Bernoulli': ba_ber
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

def generate_multidimensional_data_object_synthetic(n_students,n_tasks,n_task_per_student,  n_topics = 2 , dimension=2, probabilistic=True, number_of_tasks_per_students_is_max=False):

    edge_indices, y, student_gaussians, item_gaussians, item_difficulty = generate_multidimensional_synthetic_student_data_interactions_heterogeneous(n_students,n_tasks,n_task_per_student,  n_topics, dimension, probabilistic, number_of_tasks_per_students_is_max)
    
    data = create_data_object_synthetic_heterogeneous(n_students,n_tasks,edge_indices,y, student_gaussians, item_gaussians, item_difficulty)
    return data

def polar_to_cartesian(radius, angles):
    n = len(angles)+1
    cartesian_coordinates = np.ones(n) * radius
    for i in range(n-1):
        for j in range(i):
            cartesian_coordinates[i] *= np.sin(angles[j])
        cartesian_coordinates[i] *= np.cos(angles[i])
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

    radius = np.sqrt(2) # TODO: avoid hardcoding, give in input 
    list_theta = []
    # sampling random angles for the gaussian means
    for i in range(n_topics):
        list_theta.append(np.random.uniform(0, 2*np.pi, dimension-1)) # -1 cause one dimension is fixed by the radius
    # list_theta = [np.array(theta) for theta in [[np.pi/4],[(3/2)*np.pi - np.pi/4]]]
    list_theta = [np.array([2*np.pi*(i/n_topics)]) for i in range(n_topics)]
    gaussian_means = [torch.from_numpy(polar_to_cartesian(radius, theta)).float() for theta in list_theta]

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
    return edge_indices, y, student_gaussians, item_gaussians, item_difficulty 


def generate_multidimensional_data_object_synthetic_geometric(n_students,n_tasks,n_task_per_student,  n_topics = 2 , dimension=2, radius=0.1, probabilistic=True, number_of_tasks_per_students_is_max=False):

    edge_indices, y, ability, difficulty = generate_synthetic_geometric_interactions(n_students,n_tasks,n_task_per_student,  n_topics, dimension, radius, probabilistic, number_of_tasks_per_students_is_max)
    data = create_data_object_synthetic_geometric(n_students,n_tasks,edge_indices,y, ability, difficulty)
    return data


def generate_synthetic_geometric_interactions(
        n_students,
        n_items,
        n_tasks_per_students,
        n_topics=1, 
        dimension = 2, 
        radius = 0.1,
        probabilistic=True,
        number_of_tasks_per_students_is_max=False):
    
    assert n_topics == 1, "This function is only for 1D"

    # GENERATING NODE POSITIONS AND PLOTTING THEM 
    # NB: ALSO SETTING RADIUS USED BELOW
    student_geom = np.random.rand(n_students,dimension)
    items_geom = np.random.rand(n_items,dimension)

    # OBTAINING PROBABILITY (AND PLOTTING PATTERNS FOR ON EDGES -- POSITION TAKING AVERAGE BETWEEN NODES INVOLVED IN THE INTERACTIONS)
    label_probs = []
    for i in range(n_students):
        for j in range(n_items):
            if np.linalg.norm(student_geom[i]-items_geom[j]) < radius:
                # first coordinate used as student ability 
                # second coordinate used as item difficulty
                x = student_geom[i][0]-items_geom[j][1]
                label_probs.append(1/(1+np.exp(-x)))

    # CREATING LIST OF EDGES
    edges = []
    for i in range(n_students):
        for j in range(n_items):
            if np.linalg.norm(student_geom[i]-items_geom[j]) < radius:
                edges.append((i,j))

    # CREATING LIST OF LABELS
    labels = [1 if random.random() < p else 0 for p in label_probs]

    return edges, labels, student_geom[:,0], items_geom[:,1]



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

def create_data_object_synthetic_geometric(n_students,n_tasks,edge_indices, y, ability, difficulty):
    data  = HeteroData()

    # Save node indices
    data['student'].node_id = torch.arange(n_students)
    data['item'].node_id = torch.arange(n_tasks)

    # Save difficulty and ability
    data['student'].ability = torch.tensor(ability)
    data['item'].difficulty = torch.tensor(difficulty)
    # data['item'].difficulty = torch.tensor([item_difficulty[i] for i in range(n_tasks)])


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


