#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import scanpy as sc
import scvelo as scv
import warnings
warnings.simplefilter("ignore", category=UserWarning)
warnings.simplefilter("ignore", category=FutureWarning)
warnings.simplefilter("ignore", category=DeprecationWarning)
import anndata as ad
import numpy as np
import pandas as pd
import seaborn as sns#; sns.set()
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA

from dynamicviz.dynamicviz import boot, score

import trimap
import pacmap
#import phate



# read in SVZ 1000-cell dataset
SVZ_X = pd.read_csv("svz_data/svz_data.csv")
SVZ_X = SVZ_X.values
SVZ_X = SVZ_X/SVZ_X.sum(axis=1)[:,np.newaxis] # row normalize
SVZ_X = np.log1p(SVZ_X*10000) # scale up and take log1p

SVZ_y = np.genfromtxt("svz_data/SVZ_y.txt", dtype=str)
SVZ_age = np.genfromtxt("svz_data/SVZ_age.txt")
SVZ_Y = pd.DataFrame(np.vstack((SVZ_y,SVZ_age)).T, columns=["celltype", "age"])


# In[ ]:


# run for base methods
savebase = "hyperparameters/SVZ_n1000_"
nboots = 100 # CHANGE TO 100
njobs = 4

hyper_dict = {
    'tsne': [2,5,10,20,40,80,160,320,640], # perplexity
    'umap': [700,800,900], # n_neighbors [5,10,20,40,80,160,320,640] + [700, 800, 900]
    'lle': [20,30,40,50,60,80,100,120,140], # n_neighbors
    'mlle': [20,30,40,50,60,80,100,120,140], # n_neighbors
    'isomap': [180,220,260,300,340,380,420], # n_neighbors: [20,40,50,60,80,100,120,140] + [180,220,260,300,340,380,420]
    'trimap': [2,4,6,8,10,12,14,16,18,20], # n_inliers
    'pacmap': [2,6,10,14,18,22,26,30,34,38], # n_neighbors
}


for method in ['umap', 'isomap']:#['lle', 'mlle', 'pca', 'isomap', 'trimap', 'pacmap', 'tsne', 'umap', 'mds']:
    
    try:
    
        if method == 'pca': # no hyperparameters to optimize
            savename = savebase+'B'+str(nboots)+'_'+method
            out = boot.generate(SVZ_X, Y=SVZ_Y, method=method, B=nboots, use_n_pcs=False, num_jobs=njobs,
                save=savename+".csv", random_seed=452)
            variance_scores = score.variance(out, method="global", X_orig=SVZ_X, normalize_pairwise_distance=False)
            np.savetxt(savename+'_variance.csv', variance_scores)
            
        elif method == 'mds': # no hyperparameters to optimize
            savename = savebase+'B'+str(nboots)+'_'+method
            out = boot.generate(SVZ_X, Y=SVZ_Y, method=method, B=nboots, use_n_pcs=50, num_jobs=njobs,
                save=savename+".csv", random_seed=452)
            variance_scores = score.variance(out, method="global", X_orig=SVZ_X, normalize_pairwise_distance=False)
            np.savetxt(savename+'_variance.csv', variance_scores)

        else: # optimize hyperparameters

            for hyper in hyper_dict[method]:

                if method == 'trimap':
                    method2 = trimap.TRIMAP(n_dims=2, n_inliers=hyper)
                    savename = savebase+'B'+str(nboots)+'_trimap_neighbors'+str(hyper)
                    out = boot.generate(SVZ_X, Y=SVZ_Y, method=method2, B=nboots, use_n_pcs=50, num_jobs=njobs,
                        save=savename+".csv", random_seed=452)

                elif method == 'pacmap':
                    method2 = pacmap.PaCMAP(n_dims=2, n_neighbors=hyper)
                    savename = savebase+'B'+str(nboots)+'_pacmap_neighbors'+str(hyper)
                    out = boot.generate(SVZ_X, Y=SVZ_Y, method=method2, B=nboots, use_n_pcs=50, num_jobs=njobs,
                        save=savename+".csv", random_seed=452)

                else: # base methods

                    if method == 'tsne': # perplexity
                        savename = savebase+'B'+str(nboots)+'_tsne_perplexity'+str(hyper)
                        out = boot.generate(SVZ_X, Y=SVZ_Y, method=method, B=nboots, use_n_pcs=50, num_jobs=njobs,
                            save=savename+".csv", random_seed=452, perplexity=hyper)

                    else: # n_neighbors
                        savename = savebase+'B'+str(nboots)+'_'+method+'_neighbors'+str(hyper)
                        out = boot.generate(SVZ_X, Y=SVZ_Y, method=method, B=nboots, use_n_pcs=50, num_jobs=njobs,
                            save=savename+".csv", random_seed=452, n_neighbors=hyper)

                # Generate variance scores and save
                variance_scores = score.variance(out, method="global", X_orig=SVZ_X, normalize_pairwise_distance=False)
                np.savetxt(savename+'_variance.csv', variance_scores)
                
    except:
        print('Failed on '+method)
        continue