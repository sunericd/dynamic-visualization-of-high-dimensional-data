#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import warnings
warnings.simplefilter("ignore", category=UserWarning)
warnings.simplefilter("ignore", category=FutureWarning)
warnings.simplefilter("ignore", category=DeprecationWarning)
import numpy as np
import pandas as pd
from scipy.ndimage.filters import gaussian_filter1d
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA

from dynamicviz.dynamicviz import boot, score


# In[2]:


def sample_gaussian_mixture(n,p,magnitude=1,k=5):
    '''
    Generates a n x p array of n observations with p features from a mixture of p-dimensional unit variance Gaussians
        No covariance between Gaussians
        
    magnitude corresponding to mean (larger means further apart and better separated)
    k = number of Gaussians
    '''
    if k > p:
        raise Exception("k cannot be larger than p")
    
    X = np.zeros((n,p))
    y = np.zeros(n)
    for i in range(n):
        ri = np.random.randint(k)
        means = np.zeros(p)
        means[ri] = magnitude
        X[i,:] = np.random.multivariate_normal(means, np.identity(p), 1)
        y[i] = ri

    return (X, y)


# # Figure 2B: Empirical validation of bootstrap theory

# In[ ]:

for method in ["tsne", "umap"]:
    p = 50
    nboots=100

    n_list = []
    total_v = np.array([])
    noboot_v = np.array([])
    nostochastic_v = np.array([])

    for n in [80,160,320,640,1280,2560]:
        
        # generate random sample from mixture gaussian
        X, y = sample_gaussian_mixture(n,p,magnitude=4,k=5)
        
        
        # bootstrap variance score
        if method == "tsne":
            out = boot.generate(X, Y=y, method=method, B=nboots, use_n_pcs=False, num_jobs=4,
                    save=False, random_seed=452, init='pca')
        else:
            out = boot.generate(X, Y=y, method=method, B=nboots, use_n_pcs=False, num_jobs=4,
                save=False, random_seed=452)
        if n < 1000:
            variance_scores = score.variance(out, method="global", X_orig=X, normalize_pairwise_distance=False)
        else:
            variance_scores = score.variance(out, method="random", k=100, X_orig=X, normalize_pairwise_distance=False)
        
        
        # no bootstrap variance score
        if method == "tsne":
            out = boot.generate(X, Y=y, method=method, B=nboots, use_n_pcs=False, num_jobs=4,
                    save=False, random_seed=452, no_bootstrap=True, init='pca')
        else:
            out = boot.generate(X, Y=y, method=method, B=nboots, use_n_pcs=False, num_jobs=4,
                save=False, random_seed=452, no_bootstrap=True)
        if n < 1000:
            variance_scores1 = score.variance(out, method="global", X_orig=X, normalize_pairwise_distance=False)
        else:
            variance_scores1 = score.variance(out, method="random", k=100, X_orig=X, normalize_pairwise_distance=False)
        
        
        # no stochastic DR variance score (set random_state)
        if method == "tsne":
            out = boot.generate(X, Y=y, method=method, B=nboots, use_n_pcs=False, num_jobs=4,
                    save=False, random_seed=452, random_state=452, init='pca')
        else:
            out = boot.generate(X, Y=y, method=method, B=nboots, use_n_pcs=False, num_jobs=4,
                save=False, random_seed=452, random_state=452)
        if n < 1000:
            variance_scores2 = score.variance(out, method="global", X_orig=X, normalize_pairwise_distance=False)
        else:
            variance_scores2 = score.variance(out, method="random", k=100, X_orig=X, normalize_pairwise_distance=False)
            
        
        # append results
        n_list += [n]*n
        total_v = np.concatenate((total_v, variance_scores))
        noboot_v = np.concatenate((noboot_v, variance_scores1))
        nostochastic_v = np.concatenate((nostochastic_v, variance_scores2))


    df = pd.DataFrame(np.vstack((n_list,total_v,noboot_v,nostochastic_v)).T,
                  columns=["n", "Global variance", "Fixed data variance", "Fixed DR variance"])
    df.to_csv(method+"_N80_2560_results_totalvar.csv", index=False)