import torch
import tqdm
import numpy as np
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score, silhouette_score, pairwise_distances
import pandas as pd
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from torch.optim import Adam
from torch_geometric.data import Data, HeteroData
import torch_geometric.transforms as T
from torch.autograd import grad
from torch.autograd.functional import jacobian
from torch.nn import Linear, MSELoss
from torch_geometric.nn import to_hetero

mse = MSELoss(reduction='mean')
softmax = torch.nn.Softmax(dim=1)

@torch.no_grad()
def compute_clustering_indices(model, data, df_item, device, grouping_variable, 
                               target_variable, shuffle=False, seed=1, minsamples=10):
    scores = {
                'DB': {},
                'SH': {},
                'CH': {}
    }
    
    unique_variable = df_item[grouping_variable].dropna().unique()
    df_item = df_item.reset_index()
    
    data = data.to(device)
    try:
        pred, z_dict, _ = model(data)
    except:
        z_dict = model.get_embeddings(data)
        
    embedding = z_dict['item'].detach().cpu().numpy()   

    for category in unique_variable:
        select =  (df_item[target_variable].notnull()) & (df_item[grouping_variable] == category)
        X = embedding[select, :]
        labels = df_item[target_variable].values[select]
        
        # ensure a minimum of samples
        tab = df_item[target_variable].loc[select].value_counts()
        ind = tab[tab >= minsamples].index
        w = df_item[target_variable].loc[select].isin(ind).values
        labels = labels[w]
        X = X[w, :]
        if shuffle:
            if seed > 0:
                np.random.seed(seed)  
            np.random.shuffle(labels)
        try:
            scores['DB'][category] = 1/davies_bouldin_score(X, labels)
            scores['SH'][category] = silhouette_score(X, labels, metric='euclidean')
            scores['CH'][category] = calinski_harabasz_score(X, labels)
        except:
            scores['DB'][category] = np.nan
            scores['SH'][category] = np.nan
            scores['CH'][category] = np.nan

    return scores

@torch.no_grad()
def compute_domain_distances(model, data, df_item, device, shuffle=False, seed=1, nsamples=0):
    
    df_item = df_item.reset_index()
    
    minsamples = df_item.scale.value_counts().min()
    nsamples = min(minsamples, nsamples)
    
    # number of sampled items per scale
    sampled_df = df_item.groupby('scale', as_index=False).apply(lambda x: x.sample(np.min((len(x), nsamples)))).copy()
    print(df_item)
    print(sampled_df)
    unique_domains = df_item['domain'].dropna().unique()
    unique_scales = df_item['scale'].dropna().unique()

    try:
        pred, z_dict, _ = model(data)
        embedding = z_dict['item'].detach().cpu().numpy()
    except:
        z_dict = model.get_embeddings(data)
        embedding = z_dict['item']    
    embedding = embedding[sampled_df.index, :]
    sampled_df = sampled_df.reset_index()
    # print counts
    # print(df_item.groupby('scale').count()['index'])
    nscales = len(unique_scales)
    mean_distances = np.zeros((nscales, nscales))
    within_domain = []
    between_domain = []
    within_between_scale = np.zeros((len(unique_scales), len(unique_scales)))
    
    if shuffle:
        if seed > 0:
            np.random.seed(seed) 
        sampled_df[['scale','domain']] = sampled_df[['scale','domain']].sample(frac=1).values
        
        #np.random.shuffle(domains)
        # remap domains
        print(domain_mapping)
        domain_mapping = df_item[['scale','domain']].drop_duplicates()
        domain_mapping['domain'] = domain_mapping['domain'].sample(frac=1).values
        domain_mapping.set_index('scale')['domain'].to_dict()
        print(domain_mapping)
        sampled_df['domain'] = sampled_df['scale'].apply(lambda x: domain_mapping[x])
        
    else: 
        print(df_item.scale.value_counts())
        print(sampled_df[['scale','domain']].scale.value_counts())
    
    print(sampled_df.head(30))
    
    for i, scale_i in enumerate(unique_scales):
        for j, scale_j in enumerate(unique_scales):
               
            if i >= j:
                select_i = sampled_df.loc[sampled_df['scale'] == scale_i].index #.get_level_values(1)
                select_j = sampled_df.loc[sampled_df['scale'] == scale_j].index #.get_level_values(1)
                X_i = embedding[select_i, :]
                X_j = embedding[select_j, :]
                D = pairwise_distances(X_i, X_j)
                print(X_i.shape)
                try:
                    mean_distances[i, j] = np.mean(D)
                except:
                    print(D, scale_i, scale_j)
            else:
                continue
                    
            if i > j:
                mean_distances[j, i] = mean_distances[i, j]
                                    
            domain_i = domains[select_i][0]
            domain_j = domains[select_j][0]
            
            print(domain_i, domain_j)
            if domain_i == domain_j: 
                within_domain.append(mean_distances[i, j])
            else:
                between_domain.append(mean_distances[i, j])

    for i, scale_i in enumerate(unique_scales):
        within_between_scale[1:, i] = np.delete(mean_distances[i, :], i) 
        within_between_scale[0, i] = mean_distances[i, i] 

    #print('within_between')
    #print(within_between_scale)             
    #print('mean distances')
    #print(mean_distances)
    print("within")
    print(within_domain)
    within_domain = np.mean(within_domain)      
    print("between")
    print(between_domain)
    between_domain = np.mean(between_domain)
    

    return within_domain, between_domain, mean_distances, unique_scales, within_between_scale


@torch.no_grad()
def evaluate_items(model, data, df_item, device, shuffle=False, seed=1, minsamples=10, nsamples=500):
    
    scores_matrix = compute_clustering_indices(model, data, df_item, device, 'scale', 
                                        'matrix', shuffle=shuffle, seed=seed, 
                                        minsamples=minsamples)

    scores_topic = compute_clustering_indices(model, data, df_item, device, 'scale', 
                                        'topic', shuffle=shuffle, seed=seed, 
                                        minsamples=minsamples)

    within_domain, between_domain, mean_distances, unique_scales, within_between_scale = compute_domain_distances(model, data, df_item, device, shuffle=shuffle, seed=seed, nsamples=nsamples)
    
    return scores_matrix, scores_topic, within_domain, between_domain, mean_distances, unique_scales, within_between_scale


##################################################################
# deprecated
##################################################################

@torch.no_grad()
def clustering(model, data, df_item, device, unique_scales, unique_domains, shuffle=False, seed=1, NSAMPLES=1000):

    scores = {
                'DB': {},
                'SH': {},
                'CH': {}
    }
    
    df_item = df_item.reset_index()
    data = data.to(device)
    pred, z_dict, _ = model(data)
    #select = df_item['scale'].isin(['dles', 'ehoe', 'mzuv']).values
    embedding = z_dict['item'].detach().cpu().numpy()
    for scale in unique_scales:
        select = df_item['scale'] == scale 
        X = embedding[select, :]
        notnan = df_item['matrix'].notnull().values[select]
        labels = df_item['matrix'].values[select]
        labels = labels[notnan]
        X = X[notnan, :]
        
        if shuffle:
            if seed > 0:
                np.random.seed(seed)  
            np.random.shuffle(labels)
        try:
            scores['DB'][scale] = davies_bouldin_score(X, labels)
            scores['SH'][scale] = silhouette_score(X, labels, metric='euclidean')
            scores['CH'][scale] = calinski_harabasz_score(X, labels)
        except:
            scores['DB'][scale] = np.nan
            scores['SH'][scale] = np.nan
            scores['CH'][scale] = np.nan
            
    sampled_df = df_item.groupby('scale', as_index=False).apply(lambda x: x.sample(np.min((len(x), NSAMPLES))))
    # print counts
    # print(df_item.groupby('scale').count()['index'])
    nscales = len(unique_scales)
    mean_distances = np.zeros((nscales, nscales))
    within = []
    between = []
    
    domains = df_item['domain'].values
    
    if shuffle:
        if seed > 0:
            np.random.seed(seed)  
        np.random.shuffle(domains)

    for i, scale_i in enumerate(unique_scales):
        for j, scale_j in enumerate(unique_scales):
               
            if i >= j:
                select_i = sampled_df.loc[sampled_df['scale'] == scale_i].index.get_level_values(1)
                select_j = sampled_df.loc[sampled_df['scale'] == scale_j].index.get_level_values(1)
                X_i = embedding[select_i, :]
                X_j = embedding[select_j, :]
                D = pairwise_distances(X_i, X_j)
                mean_distances[i, j] = np.mean(D)
            else:
                continue
                    
            if i > j:
                mean_distances[j, i] = mean_distances[i, j]
               
            domain_i = domains[select_i][0]
            domain_j = domains[select_j][0]
            
            if domain_i == domain_j: 
                within.append(mean_distances[i, j])
            else:
                between.append(mean_distances[i, j])
                
    within_domain = np.mean(within)      
    between_domain = np.mean(between)      

#    for matdiff in unique_matdiff:
#        scores[matdiff] = {}
#        for scale in unique_scales:
#            select = np.logical_and(df_item['scale'] == scale, df_item['matdiff'] == matdiff)
#            X = embedding[select, :]
#            labels = df_item['matrix'].values[select]
#        
#            if shuffle:
#                np.random.shuffle(labels)
#                
#            try:
#                scores[matdiff][scale] = davies_bouldin_score(X, labels)
#            except:
#                scores[matdiff][scale] = np.nan

    return scores, within_domain, between_domain, mean_distances

