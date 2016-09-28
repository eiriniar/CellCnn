import os, errno
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import pairwise_kernels
import sklearn.utils as sku
from cellCnn.downsample import random_subsample, kmeans_subsample, outlier_subsample
from cellCnn.downsample import weighted_subsample
from  sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import fcluster
from scipy import stats
from collections import Counter
from lifelines.statistics import logrank_test


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise
        
def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]
            
def ftrans(x, c):
        return np.arcsinh(1./c * x)

def rectify(X):
    return np.max(np.hstack([X.reshape(-1,1), np.zeros((X.shape[0],1))]),
                  axis=1)

def combine_samples(data_list, sample_id, flags):
    accum_x, accum_y, accum_z = [], [], []
    for x, y, z in zip(data_list, sample_id, flags):
        accum_x.append(x)
        accum_z.append(z)
        accum_y.append(y * np.ones(x.shape[0], dtype=int))
    
    return np.vstack(accum_x), np.hstack(accum_y), np.vstack(accum_z)

def param_vector(params, regression=False):
    W, b = params['conv']
    W = np.squeeze(W)
    W_out, b_out = params['output']
    
    if regression:
        W_out = np.squeeze(W_out)
    else:
        W_out = np.squeeze(W_out)[:,1]
    
    # store the (convolutional weights + biases + output weights) per filter
    W_tot = np.hstack([np.squeeze(W),
                        b.reshape(-1,1),
                        W_out.reshape(-1,1)])
    return W_tot

def keras_param_vector(params, regression=False):
    W = np.squeeze(params[0])
    b = params[1]
    #W_out = np.diag(params[2]).reshape(-1,1)
    W_out = params[2]

    # store the (convolutional weights + biases + output weights) per filter
    W_tot = np.hstack([W, b.reshape(-1,1), W_out])
    return W_tot

def representative(data, metric='cosine', stop=None):
    if stop is None:
        i = np.argmax(np.sum(pairwise_kernels(data, metric=metric), axis=1))
    else:
        i = np.argmax(np.sum(pairwise_kernels(data[:,:stop], metric=metric), axis=1))
    return data[i]


def cluster_tightness(data, metric='cosine'):
    centroid = np.mean(data, axis=0).reshape(1,-1)
    return np.mean(pairwise_kernels(data, centroid, metric=metric))


def compute_consensus_profiles(param_dict, accuracies, accur_thres=.99,
                                regression=False, prioritize=False,
                                dendrogram_cutoff=.5):
    accum = []
    
    # if not at least 3 models reach the accuracy threshold, select the filters from the 3 best
    if np.sort(accuracies)[-3] < accur_thres:
        accur_thres = np.sort(accuracies)[-3]

    # combine filters from multiple models
    for i, params in param_dict.items():
        if accuracies[i] >= accur_thres:
            W_tot = keras_param_vector(params, regression)
            accum.append(W_tot)      
    w_strong = np.vstack(accum)
    
    # perform hierarchical clustering on cosine distances
    if w_strong.shape[0] > 1:
                    
        Z = linkage(w_strong[:,:-2], 'average', metric='cosine')
        clusters = fcluster(Z, dendrogram_cutoff, criterion='distance') - 1    
        c = Counter(clusters)
        
        # generate the consensus signatures     
        if not prioritize:
            cons = []
            for key, val in c.items():
                if val > 1:
                    members = w_strong[clusters == key]
                    cons.append(representative(members, stop=-2))
            cons_profile = np.vstack(cons)
        
        else:
            cos_scores = []
            for ckey, val in c.items():
                coeff = np.sign(np.mean(w_strong[clusters == ckey, -1]))
                cos_scores.append(coeff * np.sum(clusters == ckey))
            ckey = c.keys()[np.argmax(cos_scores)] 
            members = w_strong[clusters == ckey]
            cons_profile = representative(members, stop=-2)
           
    
        cl_res = {'w': w_strong, 'cluster_linkage': Z, 'cluster_assignments': clusters}
        
    else:
        cons_profile = np.squeeze(w_strong)
        cl_res = None

    return cons_profile, cl_res


def normalize_outliers(X, lq=.5, hq=99.5, stop=None):
    if stop is None:
        stop = X.shape[1]
    for jj in range(stop):
        marker_t = X[:,jj]
        low, high = np.percentile(marker_t, lq), np.percentile(marker_t, hq)
        X[marker_t < low, jj] = low
        X[marker_t > high, jj] = high
    return X

def normalize_outliers_to_control(ctrl_list, list2, lq=.5, hq=99.5, stop=None):
    X = np.vstack(ctrl_list)
    accum = []
    if stop is None:
        stop = X.shape[1]
    
    for xx in (ctrl_list + list2):
        for jj in range(stop):
            marker_ctrl = X[:, jj]
            low, high = np.percentile(marker_ctrl, lq), np.percentile(marker_ctrl, hq)
            marker_t = xx[:, jj]
            xx[marker_t < low, jj] = low
            xx[marker_t > high, jj] = high
        accum.append(xx)

    return accum

def compute_moments(X):
    m1 = np.mean(X, axis=-1)
    return m1

# z-transform ignoring extreme values
def correct_mean(x_list, idx_to_normalize):

    normalized_list = x_list

    for imark in idx_to_normalize:       
        for ii, val_x in enumerate(normalized_list):
            data = val_x[:,imark].copy()
            upper = np.percentile(data, 99.5)
            lower = np.percentile(data, 0.5)
            data = data[data < upper]
            data = data[data > lower]
            sc = StandardScaler()
            sc.fit(data.reshape(-1,1))
            normalized_list[ii][:,imark] = np.squeeze(sc.transform(val_x[:,imark].reshape(-1,1)))

    return normalized_list

def landmark_normalization(x_list, idx_to_normalize):
    positions = np.linspace(-1, 7, 801)
    normalized_list = x_list
    
    for imark in idx_to_normalize:      
        for ii, val_x in enumerate(normalized_list):
            kernel = stats.gaussian_kde(val_x[:,imark])
            landmark = positions[np.argmax(kernel(positions))]
            normalized_list[ii][:,imark] = val_x[:,imark] - landmark
            
    return normalized_list


## Utilities for generating random subsets ##


def filter_per_class(X, y, ylabel): 
    return X[np.where(y == ylabel)]


def per_sample_subsets(X, nsubsets, ncell_per_subset, k_init=False):
    nmark = X.shape[1]
    shape = (nsubsets, nmark, ncell_per_subset)
    Xres = np.zeros(shape)
    
    if not k_init:
        for i in range(nsubsets):
            X_i = random_subsample(X, ncell_per_subset)
            Xres[i] = X_i.T
    else:
        for i in range(nsubsets):
            X_i = random_subsample(X, 2000)
            X_i = kmeans_subsample(X_i, ncell_per_subset, random_state=i)
            Xres[i] = X_i.T

    return Xres    

        
def generate_subsets(X, pheno_map, sample_id, nsubsets, ncell, k_init=False):
    S = dict()
    n_out = len(np.unique(sample_id))
    
    for ylabel in range(n_out):
        X_i = filter_per_class(X, sample_id, ylabel)
        S[ylabel] = per_sample_subsets(X_i, nsubsets, ncell, k_init)
        
    # mix them
    data_list, y_list = [], []
    for y_i, x_i in S.items():
        data_list.append(x_i)
        y_list.append(pheno_map[y_i] * np.ones(x_i.shape[0], dtype=int)) 
         
    Xt = np.vstack(data_list)
    yt = np.hstack(y_list)  
    Xt, yt = sku.shuffle(Xt, yt) 
    return Xt, yt


def per_sample_biased_subsets(X, x_ctrl, nsubsets, ncell_final,
                            to_keep, ratio_biased):
    nmark = X.shape[1]
    Xres = np.empty((nsubsets, nmark, ncell_final))
    nc_biased = int(ratio_biased * ncell_final) 
    nc_unbiased = ncell_final - nc_biased 

    for i in range(nsubsets):
        print i
        x_unbiased = random_subsample(X, nc_unbiased)
        
        if (i % 100) == 0:
            x_outlier, outlierness = outlier_subsample(X, x_ctrl, to_keep)
    
        x_biased = weighted_subsample(x_outlier, outlierness, nc_biased)
        Xres[i] = np.vstack([x_biased, x_unbiased]).T
    
    return Xres

                        
def generate_biased_subsets(X, pheno_map, sample_id, x_ctrl, nsubset_ctrl, nsubset_biased,
                            ncell_final, to_keep, id_ctrl, id_biased):
    S = dict()
    for ylabel in id_biased:
        X_i = filter_per_class(X, sample_id, ylabel)
        n = nsubset_biased[pheno_map[ylabel]]
        S[ylabel] = per_sample_biased_subsets(X_i, x_ctrl, n,
                                             ncell_final, to_keep, 0.5)
       
    for ylabel in id_ctrl:
        X_i = filter_per_class(X, sample_id, ylabel)
        S[ylabel] = per_sample_subsets(X_i, nsubset_ctrl, ncell_final, k_init=False)

    # mix them
    data_list, y_list = [], []
    for y_i, x_i in S.items():
        data_list.append(x_i)
        y_list.append(pheno_map[y_i] * np.ones(x_i.shape[0], dtype=int)) 
         
    Xt = np.vstack(data_list)
    yt = np.hstack(y_list)  
    Xt, yt = sku.shuffle(Xt, yt) 
    return Xt, yt


def logrank_pval(stime, censor, g1):
    res = logrank_test(stime[g1], stime[~g1], censor[g1], censor[~g1], alpha=.95)
    return res.p_value
    
    
