#!/usr/bin/env python
# coding: utf-8

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
import os
import scanpy as sc

from dynamicviz.dynamicviz import boot, score

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("data", help="either 'synthetic' or 'merfish'")
parser.add_argument("method", help="specify 'umap', 'tsne', or 'pca'")
args = parser.parse_args()

save_dir = "runtime_results_"+str(args.data)
method = args.method

p = 50 # use 50 features to simulate use case of PCA preprocessing
nboots = 10
k = 50 # number of neighbors to use in random approximation of variance score
global_off = True # whether to turn off global variance score calculation

# create results folders
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
if not os.path.exists(os.path.join(save_dir, method)):
    os.makedirs(os.path.join(save_dir, method))


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

#n_values = [100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600, 51200, 102400] # first launch, oom error at 25.6k
n_values = [25600, 51200, 102400] # to run w/o global variance score
run_time_keys = ["viz_DR", "viz_align", "global_neighborhood", "global_distances", "global_normalization",
                "global_variance", "random_neighborhood", "random_distances", "random_normalization", "random_variance"]

# init run times dataframe
rt_df = pd.DataFrame(columns=["n"]+run_time_keys)


# read merfish data if needed
use_n_pcs = False
if args.data == "merfish":
    adata = sc.read_h5ad("sc_data/mouse_merfish_MOp/counts.h5ad")
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    merfish_metadata = pd.read_csv("sc_data/mouse_merfish_MOp/cell_metadata.csv")
    adata.obs = merfish_metadata
    if args.method != 'pca':
        use_n_pcs = 50


# iterate n values and save results
for n in n_values:

    new_row = [[n]]

    # generate synthetic data and save outputs
    if args.data == "synthetic":
        X, y = sample_gaussian_mixture(n,p,magnitude=4,k=5)
    elif args.data == "merfish":
        adata_sub = adata.copy()
        sc.pp.subsample(adata_sub, n_obs=n)
        X = adata_sub.X
        y = adata_sub.obs["subclass"].values
    else:
        raise Exeption("data argument not known")
    print(X.shape)
    
    # run dynamic visualization
    if method == "tsne":
        out, times1 = boot.generate(X, Y=y, method=method, B=nboots, use_n_pcs=use_n_pcs, num_jobs=4,
                save=os.path.join(save_dir, method, "out_n"+str(n)+".csv"), random_seed=452,
                return_times=True, init='pca')
    else:
        out, times1 = boot.generate(X, Y=y, method=method, B=nboots, use_n_pcs=use_n_pcs, num_jobs=4,
        save=False, random_seed=452, return_times=True)
    
    new_row.append([times1["bootstrapped_DR"]])
    new_row.append([times1["alignment_DR"]])
    
    # run global variance score
    if global_off is False:
        variance_scores, times2 = score.variance(out, method="global", X_orig=X, normalize_pairwise_distance=False,
                                                    return_times=True)
        np.savetxt(os.path.join(save_dir, method, "global_variance_n"+str(n)+".csv"), variance_scores)
        
        new_row.append([times2["neighborhood"]])
        new_row.append([times2["distances"]])
        new_row.append([times2["normalization"]])
        new_row.append([times2["variance"]])
    else:
        new_row.append([0])
        new_row.append([0])
        new_row.append([0])
        new_row.append([0])
    
    # run random approx variance score
    rand_variance_scores, times3 = score.variance(out, method="random", k=k, X_orig=X, normalize_pairwise_distance=False,
                                                return_times=True)
    np.savetxt(os.path.join(save_dir, method, "rand"+str(k)+"_variance_n"+str(n)+".csv"), rand_variance_scores)
    
    new_row.append([times3["neighborhood"]])
    new_row.append([times3["distances"]])
    new_row.append([times3["normalization"]])
    new_row.append([times3["variance"]])
    
    # save run times (update file)
    rt_df = pd.concat((rt_df, pd.DataFrame(np.array(new_row).T, columns=["n"]+run_time_keys)))
    rt_df.to_csv(os.path.join(save_dir, method+"_runtimes_to_n"+str(n)+".csv"), index=False)
        
    