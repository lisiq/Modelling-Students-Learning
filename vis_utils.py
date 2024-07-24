import torch
import os
import io
import pickle
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
from cluster_utils import compute_clustering_indices
import statsmodels.api as sm
import re
from scipy.stats import pearsonr


#dimred = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3)
dimred = PCA(whiten=False)

ALPHA = 0.3
ALPHALEVEL = 0.05
NPERMS = 100
POINTSIZE = 2
LINEWIDTH = 0.5
PERCENTILES = (0.01, 99.99)
MAX_PCS = 0 #8
FONTSCALE = 1.6
LEGEND_SIZE = 16

FIGSIZE = (2*6.4, 2*4.8) #width, height
FIGSIZE2 = (2*6.4, 4*4.8)

#FIGSIZE_STUDENTS = (10, 8)
#FIGSIZE_ITEMS = (8, 6)
#FIGSIZE_EDGES = (8, 18)
FIGSIZE_VID = (12, 6)
NFRAMES = 100
INTERVAL = 500
DPI = 1000
AGEDELTA = 0.5
AGE_THR = 2


CONTINUOUS_VARS =  ['age', 'ability', 'frequency', 'previous_sessions', 'years_from_start']

COMP_NAMES = ['x','y','z','u','v']
# labels
COMP_LABELS = {comp: f'Component {i}' for i, comp in enumerate(COMP_NAMES)}

CLUSTER_LABELS = {'CH': 'Calinsky-Harabasz Index', 'DB': 'Inverse of Davies-Bouldin Index'}
COMPETENCE_LABELS = {'matrix': 'Competences', 'topic': 'Topics'}

FEATURE_LABELS = {'ability': 'Ability', 'Gender': 'Gender', 'age':'Age', 
                  'Gender_motherTongue': 'Gender x Mother Tongue', 'grade':'Grade',
                  'motherTongue': 'Mother Tongue', 
                  'scale':'Competence domain', 
                  'domain':'Subject Domain',
                  'frequency':'Frequency', 
                  'previous_sessions': 'Previous Sessions', 
                  'years_from_start': 'Years of Use'}

DOMAIN_LABELS = {'d': 'German', 'e': 'English', 'f': 'French', 'm': 'Mathematics'}

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)


import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def parallel_analysis(shapeMatrix, nperms=1000):
    normalized_shapeMatrix=(shapeMatrix-shapeMatrix.mean(axis=0))/shapeMatrix.std(axis=0)
    print(shapeMatrix.mean(axis=0))
    print(shapeMatrix.std(axis=0))
    pca = PCA(shapeMatrix.shape[1]-1)
    pca.fit(normalized_shapeMatrix)
    transformedShapeMatrix = pca.transform(normalized_shapeMatrix)
    #np.savetxt("pca_data.csv", pca.explained_variance_, delimiter=",")

    random_eigenvalues = np.zeros(shapeMatrix.shape[1]-1)
    for i in range(nperms):
        random_shapeMatrix = pd.DataFrame(np.random.normal(0, 1, [shapeMatrix.shape[0], shapeMatrix.shape[1]]))
        pca_random = PCA(shapeMatrix.shape[1]-1)
        pca_random.fit(random_shapeMatrix)
        transformedRandomShapeMatrix = pca_random.transform(random_shapeMatrix)
        random_eigenvalues = random_eigenvalues+pca_random.explained_variance_ratio_
    random_eigenvalues = random_eigenvalues / nperms
    
    PC_values = np.arange(pca.n_components_) + 1
    plt.plot(PC_values, pca.explained_variance_ratio_, '--bo', label='pca-data')
    plt.plot(PC_values, random_eigenvalues, '--rx', label='pca-random')
    plt.legend()
    plt.title('Parallel analysis plot')
    plt.show()

class Results:
    def __init__(self):
        self.stats_dict = {}
        self.figs_dict = {}
    
    def add_stats(self, tag, X, Y):
        X = sm.add_constant(X)
        model = sm.OLS(Y,X, missing='drop')
        results = model.fit()
        self.stats_dict[tag] = results

    def add_fig(self, tag, axes):
        self.figs_dict[tag] = axes

    def output_stats(self, filename='./stats/stats.txt'): 
        #print(self.stats_dict)
        f = open(filename, "w")    

        for tag, results in self.stats_dict.items():        
            #print("\n" + 100*"*" + "\n")
            #print(tag)
            #print("\n" + 100*"*" + "\n")
            #print(results.summary())
            f.write("\n" + 100*"*" + "\n")
            f.write(tag)
            f.write("\n" + 100*"*" + "\n")
            f.write(results.summary().as_text())
            f.write("\n" + 100*"*")
            #print("Parameters: ", results.params)
            #print("Parameters: ", results.params)
            #print("R2: ", results.rsquared)
        f.close()

    def get_stats(self): 
        return self.stats_dict

    def get_figs(self): 
        return self.figs_dict

myresults = Results()

def save_plot(data, var, title, figname, x, y=None, plot_type='sct', equal_axes=False, palette=None, with_legend=False):

    os.makedirs(f'./vis/{figname}', exist_ok=True)
    
    fig = plt.figure()

    xlim = np.percentile(data[x], np.array(PERCENTILES))    
 
    if plot_type == 'sct':
        ylim = np.percentile(data[y], np.array(PERCENTILES))
        axes = sns.scatterplot(data=data, x=x, y=y, hue=var, s=POINTSIZE, alpha=ALPHA, palette=palette) 
        axes.set_ylim(ylim)
        axes.set_ylabel(COMP_LABELS[y])
        axes.set_xlim(xlim)
        axes.set_xlabel(COMP_LABELS[x])
        axes.legend(title=title)        
        axes.set_title(title)
        
    if plot_type == 'kde':
        axes = sns.kdeplot(data=data, x=x, hue=var, common_norm=False)
        axes.set_xlim(xlim)
        axes.set_xlabel(COMP_LABELS[x])
        axes.legend_.set_title(None)
        handles, labels = axes.get_legend_handles_labels()
        axes.legend(handles=handles[1:], labels=labels[1:])
        #axes.legend(title=title)
               
    if plot_type == 'reg':
        axes = sns.regplot(data=data, x=var, y=x, scatter_kws={'s':POINTSIZE, 'alpha':ALPHA}, line_kws={"color": "red"})
        axes.set_ylim(xlim)
        axes.set_ylabel(COMP_LABELS[x])
        axes.set_xlabel(title)
        xx = data[var].values        
        yy = data[x].values
        correlation, p_value = pearsonr(xx[~np.isnan(xx)], yy[~np.isnan(xx)])
        axes.text(0.9, 0.95, f"r={correlation:.3f} (p={p_value:.3f})", ha="right", va="top", transform = axes.transAxes)
        
    
    axes.legend(prop = { 'size': LEGEND_SIZE })
    
    if not with_legend:
        axes.legend_.remove()
    
    if equal_axes: 
        axes.set_aspect('equal', 'box')

    fig.tight_layout()
    plt.savefig(f'./vis/{figname}/type_{plot_type}-{x}-{y}-var_{var}.png', dpi=DPI)
    plt.close()
    
@torch.no_grad()
def visualize_students(model, data, device, df_student, OUTNAME, dims=('x', 'y'), equal_axes=False, encoded=True):

    data = data.to(device)
    try:
        pred, z_dict, z_edge = model(data)
    except:
        z_dict = model.get_embeddings(data, encoded=encoded)
    embedding = z_dict['student'].detach().cpu().numpy()

    dimred.fit(embedding)
    low_dim = dimred.transform(embedding)

    X = df_student
    X['x'] = low_dim[:, 0]
    X['y'] = low_dim[:, 1]
    X['z'] = low_dim[:, 2]    
    X['u'] = low_dim[:, 3]
    X['v'] = low_dim[:, 4]


    X['Gender'] = X['Gender'].apply(lambda x: 'Male' if x == 1 else 'Female')
    X['motherTongue'] = X['motherTongue'].apply(lambda x: 'German' if x == 1 else 'Other')
    X['Gender_motherTongue'] = X['Gender'].str.cat(X['motherTongue'], sep =", ")
    
    if encoded:
        suffix = ''    
    else:
        suffix = '_notenc'    
        
    figname = f'{OUTNAME}_students{suffix}'    
    save_plot(X, 'Gender', 'Gender', figname, x='x', y='y', plot_type='sct', equal_axes=equal_axes, with_legend=True)
    save_plot(X, 'motherTongue', 'Mother Tongue', figname, x='x', y='y', plot_type='sct', equal_axes=equal_axes, with_legend=True)
    save_plot(X, 'Gender_motherTongue', 'Gender x Mother Tongue', figname, x='x', y='y', plot_type='sct', equal_axes=equal_axes, with_legend=True)
    save_plot(X, 'age', 'Age', figname, x='x', y='y', plot_type='sct', equal_axes=equal_axes, palette='viridis')
    save_plot(X, 'grade', 'Grade', figname, x='x', y='y', plot_type='sct', equal_axes=equal_axes, palette='viridis')

    save_plot(X, 'Gender', 'Gender', figname, x='z', y='y', plot_type='sct', equal_axes=equal_axes, with_legend=True)
    save_plot(X, 'motherTongue', 'Mother Tongue', figname, x='z', y='y', plot_type='sct', equal_axes=equal_axes, with_legend=True)
    save_plot(X, 'Gender_motherTongue', 'Gender x Mother Tongue', figname, x='z', y='y', plot_type='sct', equal_axes=equal_axes, with_legend=False)
    save_plot(X, 'age', 'Age', figname, x='z', y='y', plot_type='sct', equal_axes=equal_axes, palette='viridis')
    save_plot(X, 'grade', 'Grade', figname, x='z', y='y', plot_type='sct', equal_axes=equal_axes, palette='viridis')      

    figname = f'{OUTNAME}_dim_students{suffix}'
    for i, mydim in enumerate(COMP_NAMES):
        save_plot(X, 'Gender', 'Gender', figname, x=mydim, plot_type='kde')
        save_plot(X, 'motherTongue', 'Mother Tongue', figname, x=mydim, plot_type='kde')
        save_plot(X, 'Gender_motherTongue', 'Gender x Mother Tongue', figname, x=mydim, plot_type='kde')
        save_plot(X, 'age', 'Age', figname, x=mydim, plot_type='reg')
        save_plot(X, 'grade', 'Grade', figname, x=mydim, plot_type='reg')

        myresults.add_stats('student_age_%s'%mydim, X['age'], X[mydim])
        myresults.add_stats('student_grade_%s'%mydim, X['grade'], X[mydim])

    fig = plt.figure()
    PC_values = np.arange(dimred.n_components_) + 1
    plt.plot(PC_values, dimred.explained_variance_ratio_*100, 'o-', linewidth=2, color='blue')
    if MAX_PCS > 0:
        plt.xticks(PC_values[:MAX_PCS])
        plt.xlim(0, max(PC_values[:MAX_PCS])+1)
    plt.title('Scree Plot')
    plt.xlabel('Principal Component')
    plt.ylabel('Variance Explained (%)')        
    fig.tight_layout()
    plt.savefig(f'./vis/{OUTNAME}_students_PCA{suffix}.png', dpi=DPI)
    #myresults.add_fig('students_PCA', plt.gca())
        
@torch.no_grad()
def visualize_items(model, data, device, df_item, OUTNAME, dims=('x', 'y'), equal_axes=False, encoded=True):

    if encoded:
        suffix = ''    
    else:
        suffix = '_notenc'    
        
    data = data.to(device)
    try:
        pred, z_dict, z_edge = model(data)
    except:
        z_dict = model.get_embeddings(data, encoded=encoded)        
    embedding = z_dict['item'].detach().cpu().numpy()

    fig = plt.figure()
    parallel_analysis(embedding)
    
    dimred.fit(embedding)
    low_dim = dimred.transform(embedding)

    X = df_item #.loc[select, :]
    
    X['domain'] = X['domain'].apply(lambda x: DOMAIN_LABELS[x])
    
    X['x'] = low_dim[:, 0]
    X['y'] = low_dim[:, 1]
    X['z'] = low_dim[:, 2]
    X['u'] = low_dim[:, 3]
    X['v'] = low_dim[:, 4]
    figname = f'{OUTNAME}_items{suffix}'
    save_plot(X, 'domain', 'Subject Domain', figname, x='x', y='y', plot_type='sct', equal_axes=equal_axes, with_legend=True)
    save_plot(X, 'scale', 'Competence Domain', figname, x='x', y='y', plot_type='sct', equal_axes=equal_axes)
    save_plot(X, 'IRT_difficulty', 'Difficulty', figname, x='x', y='y', plot_type='sct', equal_axes=equal_axes, palette='viridis')
    save_plot(X, 'IRT1_difficulty', 'Difficulty', figname, x='x', y='y', plot_type='sct', equal_axes=equal_axes, palette='viridis')
    save_plot(X, 'IRT1_discrimination', 'Discrimination', figname, x='x', y='y', plot_type='sct', equal_axes=equal_axes, palette='viridis')
    save_plot(X, 'IRT1_discrimination_transf', 'Discrimination (transformed)', figname, x='x', y='y', plot_type='sct', equal_axes=equal_axes, palette='viridis')

    save_plot(X, 'domain', 'Subject Domain', figname, x='z', y='y', plot_type='sct', equal_axes=equal_axes, with_legend=False)
    save_plot(X, 'scale', 'Competence Domain', figname, x='z', y='y', plot_type='sct', equal_axes=equal_axes)
    save_plot(X, 'IRT_difficulty', 'Difficulty', figname, x='z', y='y', plot_type='sct', equal_axes=equal_axes, palette='viridis')
    save_plot(X, 'IRT1_difficulty', 'Difficulty', figname, x='z', y='y', plot_type='sct', equal_axes=equal_axes, palette='viridis')
    save_plot(X, 'IRT1_discrimination', 'Discrimination', figname, x='z', y='y', plot_type='sct', equal_axes=equal_axes, palette='viridis')
    save_plot(X, 'IRT1_discrimination_transf', 'Discrimination (transformed)', figname, x='z', y='y', plot_type='sct', equal_axes=equal_axes, palette='viridis')

    figname = f'{OUTNAME}_dim_items{suffix}'
    for i, mydim in enumerate(COMP_NAMES):
        save_plot(X, 'domain', 'Subject Domain', figname, x=mydim, plot_type='kde')
        save_plot(X, 'scale', 'Competence Domain', figname, x=mydim, plot_type='kde')
        save_plot(X, 'IRT_difficulty', 'Difficulty', figname, x=mydim, plot_type='reg')
        save_plot(X, 'IRT1_difficulty', 'Difficulty', figname, x=mydim, plot_type='reg')
        save_plot(X, 'IRT1_discrimination', 'Discrimination', figname, x=mydim, plot_type='reg')
        save_plot(X, 'IRT1_discrimination_transf', 'Discrimination (transformed)', figname, x=mydim, plot_type='reg')
        myresults.add_stats('item_difficulty_%s'%mydim, X['IRT_difficulty'], X[mydim])
        myresults.add_stats('item_difficulty1_%s'%mydim, X['IRT1_difficulty'], X[mydim])
        myresults.add_stats('item_discrimination1_%s'%mydim, X['IRT1_discrimination'], X[mydim])
        
    fig = plt.figure()
    PC_values = np.arange(dimred.n_components_) + 1
    #plt.sca(axes[1, 1])
    plt.plot(PC_values, dimred.explained_variance_ratio_*100, 'o-', linewidth=2, color='blue')
    plt.plot(PC_values, np.cumsum(dimred.explained_variance_ratio_)*100, 'o-', linewidth=2, color='red')
    fig.legend(labels=['Variance', 'Cumulative variance'], loc='center', fontsize=10)
    
    print(dimred.explained_variance_ratio_*100)
    print(np.cumsum(dimred.explained_variance_ratio_)*100)
    
    if MAX_PCS > 0:
        plt.xticks(PC_values[:MAX_PCS])
        plt.xlim(0, max(PC_values[:MAX_PCS])+1)
    plt.title('Scree Plot')
    plt.xlabel('Principal Component')
    plt.ylabel('Variance Explained (%)')
    fig.tight_layout()
    
    plt.savefig(f'./vis/{OUTNAME}_items_PCA{suffix}.png', dpi=DPI)
    #myresults.add_fig('items_PCA', plt.gca())
        
        
@torch.no_grad()        
def visualize_edges_age(model, data, edge_indices, device, df, OUTNAME, dims=('x', 'y'), equal_axes=False, age_window=(0,100), age_lim=(8, 18), aggregate=False, with_lines=True, hue_label='age', AGE_THR=0, save=True):
    
    if hue_label not in df.columns:
        return
    
    labcols = ['studentId', 'age']
    aggcols = ['studentId', 'age']
    
    if hue_label in CONTINUOUS_VARS:
        palette = 'viridis'
    else:
        palette = 'tab10'
        aggcols = ['studentId', 'age'] + [hue_label]
        
    if hue_label != 'age':
        labcols = labcols + [hue_label]
        
    #print(age_window)
    data = data.to(device)
    pred, z_dict, z_edge = model(data)

    edge_indices = np.array(edge_indices)
    X = df.loc[edge_indices, :][labcols].reset_index()
    embedding = z_edge.detach().cpu().numpy()    
    
    if aggregate:
        cols = [ f"em_{i}" for i in range(embedding.shape[1]) ]
        embed_df = pd.DataFrame(embedding, columns=cols)
        XX = pd.concat((X, embed_df), axis=1)
        cols_ = cols
        if hue_label in CONTINUOUS_VARS[1:]:
            cols_ = cols_ + [hue_label]
        XX = XX.groupby(aggcols).agg({col: np.nanmean for col in cols_}).reset_index()
        embedding = XX[cols].values
        X = XX[[ x for x in XX.columns if x not in cols ]].copy()

    dimred.fit(embedding)
    low_dim = dimred.transform(embedding)

    age_ranges = X.groupby('studentId').age.apply(lambda x: x.max() - x.min())
    age_ranges_select = age_ranges[age_ranges > AGE_THR].index
    #print(age_ranges_select.shape)
    #print(X.shape)
    X['x'] = low_dim[:, 0]
    X['y'] = low_dim[:, 1]
    X['z'] = low_dim[:, 2]
    X['u'] = low_dim[:, 3]
    X['v'] = low_dim[:, 4]


    figname = f'{OUTNAME}_dim_edges'
    if not aggregate:
        for i, mydim in enumerate(COMP_NAMES):
            if hue_label in CONTINUOUS_VARS:
                myresults.add_stats('student_%s_%s'%(hue_label, mydim), X[hue_label], X[mydim])
                save_plot(X, hue_label, FEATURE_LABELS[hue_label], figname, x=mydim, plot_type='reg')
            else: 
                save_plot(X, hue_label, FEATURE_LABELS[hue_label], figname, x=mydim, plot_type='kde')

    lims = {}
    lims['x'] = np.percentile(X['x'], np.array(PERCENTILES))
    lims['y'] = np.percentile(X['y'], np.array(PERCENTILES))
    lims['z'] = np.percentile(X['z'], np.array(PERCENTILES))
    
    X_long = X.loc[df.studentId.isin(age_ranges_select), :]
    X_long = X_long.loc[(X_long.age >= age_window[0]) & (X_long.age <= age_window[1]), :]
    #print(X_long.shape)
    
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE2)
    if with_lines: 
        X_0 = X_long.sort_values(['studentId', 'age']).reset_index(drop=True)
        
        if False: # hue_label == 'age':
            X_1 = X_0.shift(-1)
            X_0['age_unit'] = X_0.age
            X_1['age_unit'] = X_0.age
            X_0['unit'] = X_0.index
            X_1['unit'] = X_1.index
            X_ = pd.concat((X_0, X_1), axis = 0).dropna()
            X_ = X_.sort_values(['studentId', 'age_unit']).reset_index(drop=True)

            sns.lineplot(ax=axes[0], data=X_, x='x', y='y', units='unit', estimator=None, 
                         linewidth=LINEWIDTH, alpha=ALPHA, hue='age_unit', palette=palette, legend=False, 
                         hue_norm=age_lim)    
            sns.lineplot(ax=axes[1], data=X_, x='z', y='y', units='unit', estimator=None, 
                         linewidth=LINEWIDTH, alpha=ALPHA, hue='age_unit', palette=palette, legend=False, 
                         hue_norm=age_lim)    
        #else:
            sns.lineplot(ax=axes[0], data=X_0, x='x', y='y', units='studentId', estimator=None, 
                         linewidth=LINEWIDTH, alpha=ALPHA, hue=hue_label, palette=palette, legend=False)    
            sns.lineplot(ax=axes[1], data=X_0, x='z', y='y', units='studentId', estimator=None, 
                         linewidth=LINEWIDTH, alpha=ALPHA, hue=hue_label, palette=palette, legend=False)
            
        X_1 = X_0.shift(-1)
        X_0['age_unit'] = X_0.age
        X_1['age_unit'] = X_0.age
        X_0['unit'] = X_0.index
        X_1['unit'] = X_1.index
        X_ = pd.concat((X_0, X_1), axis = 0).dropna()
        X_ = X_.sort_values(['studentId', 'age_unit']).reset_index(drop=True)
        #print(X_0)

        if hue_label == 'age':
            hue = 'age_unit'
            hue_norm = age_lim
        elif hue_label == 'ability':
            hue = 'ability'
            hue_norm = (np.min(X_0.ability), np.max(X_0.ability)) #np.percentile(X_0.ability, np.array(0.1, 99.9))
        else:
            hue = hue_label
            hue_norm = None

        sns.lineplot(ax=axes[0], data=X_, x='x', y='y', units='unit', estimator=None, 
                     linewidth=LINEWIDTH, alpha=ALPHA, hue=hue, palette=palette, legend=False, 
                     hue_norm=hue_norm)    
        sns.lineplot(ax=axes[1], data=X_, x='z', y='y', units='unit', estimator=None, 
                     linewidth=LINEWIDTH, alpha=ALPHA, hue=hue, palette=palette, legend=False, 
                     hue_norm=hue_norm)
        
    age_window = np.round(age_window[0], 2), np.round(age_window[1], 2) 
    
    if hue_label == 'age':
        hue_norm = age_lim
    else:
        hue_norm = None

    sns.scatterplot(ax=axes[0], data=X_long, x='x', y='y', hue=hue_label, 
                    palette=palette, hue_norm=hue_norm, s=POINTSIZE, legend=False, alpha=ALPHA) 
    #axes[0].text(PERCENTILES[0], PERCENTILES[1], f'Age: {age_window}' , ha='left', va='top', transform=axes[0].transAxes)
    sns.scatterplot(ax=axes[1], data=X_long, x='z', y='y', hue=hue_label, 
                    palette=palette, hue_norm=hue_norm, s=POINTSIZE, legend=False, alpha=ALPHA)   
    #axes[1].text(PERCENTILES[0], PERCENTILES[1], f'Age: {age_window}' , ha='left', va='top', transform=axes[1].transAxes)
    axlist = [0, 1]
    titlelist = [hue_label]*2
    [ axes[z].set_title(titlelist[i]) for i, z in enumerate(axlist)]
    axes[0].set_xlim(lims['x'])
    [ axes[z].set_ylim(lims['y']) for z in axlist]
    axes[1].set_xlim(lims['z'])
    
    axes[0].set_xlabel(COMP_LABELS['x'])
    [ axes[z].set_ylabel(COMP_LABELS['y']) for z in axlist]
    axes[1].set_xlabel(COMP_LABELS['z'])

    if equal_axes: 
        [ axes[z].set_aspect('equal', 'box') for z in axlist]

    #axes.set_title(title)
    #axes.legend(title=title)
    figname = f'{OUTNAME}_{hue_label}_edges'

    if aggregate: 
        figname += '_agg'           
    if with_lines: 
        figname += '_wl'   
        
    fig.tight_layout()
    if save:
        plt.savefig(f'./vis/{figname}.png', dpi=DPI)
        plt.close()

@torch.no_grad()
def visualize_edges(model, data, edge_indices, device, df, OUTNAME, **kwargs):
    
    data = data.to(device)
    
    try:
        pred, z_dict, z_edge = model(data)
        embedding = z_edge.detach().cpu().numpy()
    except:
        # not implemented
        pass
    
    dimred.fit(embedding)
    low_dim = dimred.transform(embedding)
    
    visualize_edges_age(model, data, edge_indices, device, df, OUTNAME, hue_label='frequency', **kwargs)
    visualize_edges_age(model, data, edge_indices, device, df, OUTNAME, hue_label='previous_sessions', **kwargs)
    visualize_edges_age(model, data, edge_indices, device, df, OUTNAME, hue_label='years_from_start', **kwargs)
    
    visualize_edges_age(model, data, edge_indices, device, df, OUTNAME, hue_label='age', **kwargs)
    visualize_edges_age(model, data, edge_indices, device, df, OUTNAME, hue_label='ability', **kwargs)
    visualize_edges_age(model, data, edge_indices, device, df, OUTNAME, hue_label='Gender', **kwargs)
    visualize_edges_age(model, data, edge_indices, device, df, OUTNAME, hue_label='motherTongue', **kwargs)
    visualize_edges_age(model, data, edge_indices, device, df, OUTNAME, hue_label='scale', **kwargs)
    visualize_edges_age(model, data, edge_indices, device, df, OUTNAME, hue_label='domain', **kwargs)
    
    fig = plt.figure()
    PC_values = np.arange(dimred.n_components_) + 1
    #plt.sca(axes[3, 0])
    plt.plot(PC_values, dimred.explained_variance_ratio_, 'o-', linewidth=2, color='blue')
    plt.xticks(PC_values[:MAX_PCS])
    plt.xlim(0, max(PC_values[:MAX_PCS])+1)
    plt.title('Scree Plot')
    plt.xlabel('Principal Component')
    plt.ylabel('Variance Explained')
    fig.tight_layout()
    plt.savefig(f'./vis/{OUTNAME}_edges_PCA.png', dpi=DPI)

def get_range_means(s):
    l = re.findall(r"-?\d+\.?\d*", s)
    l = [ float(x) for x in l]
    m = np.mean(l)
    return m

def plot_clustering(grouping_variable, target_variable, model, data, df_item, device, OUTNAME, minsamples=30, nperms=NPERMS, 
                    minlevels=10, encoded=True):

    scores_dict = {'CH': [], 'DB':[], 'N_LEVELS':[], 'N_SAMPLES':[]}

    for perm in range(nperms):
        #print(perm)
        scores = compute_clustering_indices(model, data, df_item, device, grouping_variable, 
                                            target_variable, shuffle=perm>0, seed=0, minsamples=minsamples, encoded=encoded)
        [ scores_dict[key].append(scores[key]) for key in scores_dict]

    for index in ['CH', 'DB']:
        scores_df = pd.DataFrame(scores_dict[index])
        
        scores_df['perm'] = scores_df.index     
        scores_df = pd.melt(scores_df, id_vars='perm', value_name='index', var_name=grouping_variable)

        scores_df['N_LEVELS'] = scores_df[grouping_variable].apply(lambda x: scores_dict['N_LEVELS'][0][x])
        scores_df['N_SAMPLES'] = scores_df[grouping_variable].apply(lambda x: scores_dict['N_SAMPLES'][0][x])

        fig = plt.subplots(ncols=2, nrows=1, figsize=FIGSIZE)
        
        ax = plt.subplot(1, 2, 1)
        scores_df['N_LEVELS'].hist(ax=ax, bins=30)
        ax = plt.subplot(1, 2, 2)
        scores_df['N_SAMPLES'].hist(ax=ax, bins=30)        
        plt.show()
        plt.close()
        scores_df = scores_df.loc[scores_df['N_LEVELS'] >= minlevels]
        print(scores_df.shape)
        if len(scores_df) == 0:
            continue
        scores_df['random'] = 'Observed data'
        scores_df.loc[ scores_df['perm'] > 0, 'random'] = 'Shuffled data'
        scores_df['scale'] = scores_df['scalexdifficulty'].str.split().str[0]
        scores_df['difficulty_bin'] = scores_df['scalexdifficulty'].str.split().str[1:].str.join(' ')
        scores_df['difficulty_num'] = scores_df['difficulty_bin'].apply(get_range_means)
        scores_df['difficulty_jit'] = scores_df['difficulty_num'] + np.random.normal(0, 0.05, size=len(scores_df['difficulty_num'])) 
        scores_df_ = scores_df.dropna().copy()
        #print(scores_df_)
        print(scores_df.shape)
        unique_scales = scores_df_['scale'].unique()
        ncols = (len(unique_scales)+1)//2
        fig, axes = plt.subplots(ncols=ncols, nrows=2, figsize=FIGSIZE, sharex=True, sharey=True)
        print(unique_scales)
        for i, scale in enumerate(unique_scales):
            ax = plt.subplot(2, ncols, i+1)
            print(i, scale)
            scores_df = scores_df_.loc[ scores_df_['scale'] == scale, :]
            maxscore_df = scores_df.groupby(['perm','random','difficulty_num'])['index'].max().reset_index()
            thr_df = scores_df.loc[scores_df.random == 'Shuffled data'].groupby(['difficulty_num'])['index'].quantile(1-ALPHALEVEL).reset_index()
            if len(maxscore_df) == 0:
                continue
            vals = maxscore_df.loc[maxscore_df.random == 'Shuffled data']['index']
            thr_global = np.quantile(vals, 1 - ALPHALEVEL)
            sns.lineplot(ax=ax, data=thr_df, #.sort_values('index'), 
                         x='difficulty_num', 
                         y='index', # grouping variable
                         color='red', linestyle='--', size=0.5)
    #            sns.scatterplot(ax=ax, data=scores_df.query('`random` == "Shuffled data"'), x=grouping_variable, y='index',
    #                            hue='random', s=3, alpha=0.5) # + ' ' + target_variable + ' ' + index)
            sns.scatterplot(ax=ax, data=scores_df.query('`random` == "Shuffled data"'), 
                          x='difficulty_jit', 
                          y='index',
                          hue='random', 
                          s=3, alpha=0.7) # + ' ' + target_variable + ' ' + index)
            sns.lineplot(ax=ax, data=scores_df.query('`random` == "Observed data"'), #.sort_values('index'), 
                         x='difficulty_num', 
                         y='index', # grouping variable
                         color='black',
                         size=0.5)
            sns.scatterplot(ax=ax, data=scores_df.query('`random` == "Observed data"'), #.sort_values('index'), 
                         x='difficulty_num', 
                         y='index', # grouping variable
                         color='black',
                         size=1)

            ax.set_title(scale)
            ax.set_xlabel('Difficulty')
            ax.set_ylabel('Cluster Validity Index')
            ax.axhline(thr_global, color = 'red', linewidth=0.5)
            #ax.set(xticklabels=[])
            ax.legend_.remove()
            ax.label_outer()
        if index == 'CH':
            fig.legend( labels=['Significance threshold \n(uncorrected)',
                                '_', 'Observed data', '_', 'Shuffled data', 
                                '_', 
                                '_', '_', '_', '_', 'Significance threshold \n(corrected)'], 
                       loc=(0.72, 0.72), fontsize=LEGEND_SIZE) #loc=(0.84, 0.2)
        
        #ax = plt.subplot(2, ncols, ncols*2)
        #ax.axis('off')

        fig.tight_layout()
        plt.savefig(f'./vis/{OUTNAME}_{grouping_variable}_{target_variable}_clustering_{index}.png')
        plt.close()
    
def doanim(HUELABEL, model, data, device, edge_indices, df, OUTNAME, EQUAL_AXES = False, html=True):
    
    MINAGE, MAXAGE = np.percentile(df.age, np.array([1, 99])) #df.age.min(), df.age.max() #
    #MAXAGE = MAXAGE 
    AGERANGE= MAXAGE - MINAGE - AGEDELTA

    def init():
        pass
        visualize_edges_age(model, data, edge_indices, 
                        device, axes, df, OUTNAME, equal_axes=EQUAL_AXES, 
                        age_window=(MINAGE, MINAGE+AGEDELTA),
                        age_lim=(MINAGE, MAXAGE),
                        aggregate=True,
                        hue_label = HUELABEL
                       )             
        #fig.tight_layout()

    def update(frame):
        [ax.clear() for ax in axes]
        visualize_edges_age(model, data, edge_indices, 
                            device, axes, df, OUTNAME, 
                            equal_axes=EQUAL_AXES, 
                            age_window=(MINAGE + frame*AGERANGE/NFRAMES, MINAGE + frame*AGERANGE/NFRAMES + AGEDELTA), 
                            age_lim=(MINAGE, MAXAGE),
                            aggregate=True,
                            hue_label = HUELABEL
                   )        
        #fig.tight_layout()
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_VID)
    ani = FuncAnimation(fig, update, frames=NFRAMES, init_func=init, blit=False, interval=INTERVAL)
    
    return ani


