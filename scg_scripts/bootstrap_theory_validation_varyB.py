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

for method in ["umap", "tsne", "pca"]:
    n_pairs = 20
    p = 50
    n=1000

    n_list = []
    boot_dist_vars = []
    real_dist_vars = []

    for nboots in [5,10,20,40,80,160,320]:
        
        # generate pairs of i and j
        Xp, yp = sample_gaussian_mixture(n_pairs*2,p,magnitude=4,k=5)
        
        for pidx in range(n_pairs):
            i = Xp[2*pidx,:]
            j = Xp[2*pidx+1,:]
            
            # generate context data
            X, y = sample_gaussian_mixture(n,p,magnitude=4,k=5)
            
            # add i and j to data
            X = np.vstack((np.vstack((X, i)), j))
            ii = n
            ji = n+1
            
            # Compute bootstrap pairwise distance variances of i and j
            embeddings, boot_idxs_list = boot.bootstrap(X, method=method, B=nboots, sigma_noise=None, no_bootstrap=False, 
                                                                        random_seed=452, num_jobs=4, use_n_pcs=False)
            
            dists = []
            
            for b in range(nboots):
                if (ii in boot_idxs_list[b]) and (ji in boot_idxs_list[b]):
                    for ib in np.argwhere(boot_idxs_list[b] == ii):
                        emb_i = embeddings[b][ib[0],:]
                        for jb in np.argwhere(boot_idxs_list[b] == ji):
                            emb_j = embeddings[b][jb[0],:]
                            
                            # compute L2 distance
                            dist_ij = np.linalg.norm(emb_i-emb_j)
                            dists.append(dist_ij)
                            
            # compute variance and append results
            if len(dists) > 1:
                n_list.append(n)
                boot_dist_vars.append(np.nanvar(dists) / np.nanmean(dists)**2)
                
                # compute real pairwise distance variances of i and j
                real_dists = []
                for b in range(nboots):
                    # generate context data
                    X, y = sample_gaussian_mixture(n,p,magnitude=4,k=5)
                    # add i and j to data
                    X = np.vstack((np.vstack((X, i)), j))
                    # Compute bootstrap pairwise distance variances of i and j
                    if method in ["tsne", "umap"]:
                        emb, boot_idxs = boot.run_one_bootstrap(X, method=method, sigma_noise=None, no_bootstrap=True,
                                    random_seeded_sequence=np.array([452]), b=0, use_n_pcs=False, n_jobs=-1)
                    else:
                        emb, boot_idxs = boot.run_one_bootstrap(X, method=method, sigma_noise=None, no_bootstrap=True,
                                    random_seeded_sequence=np.array([452]), b=0, use_n_pcs=False)
                    # get results and append
                    for ib in np.argwhere(boot_idxs == ii):
                        emb_i = emb[ib[0],:]
                        for jb in np.argwhere(boot_idxs == ji):
                            emb_j = emb[jb[0],:]
                            # compute L2 distance
                            dist_ij = np.linalg.norm(emb_i-emb_j)
                            real_dists.append(dist_ij)
                            
                real_dist_vars.append(np.nanvar(real_dists) / np.nanmean(real_dists)**2)


    # In[ ]:


    df = pd.DataFrame(np.vstack((n_list,boot_dist_vars,real_dist_vars)).T,
                      columns=["n", "Bootstrap sample distance variance", "Real sample distance variance"])
    df.to_csv(method+"_B5_320_results_.csv", index=False)