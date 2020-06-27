""" Copyright 2016-2017 ETH Zurich, Eirini Arvaniti and Manfred Claassen.

This module contains functions for downsampling.

"""

import logging
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils.extmath import row_norms
from sklearn.utils import check_random_state
from sklearn.metrics.pairwise import pairwise_distances

logger = logging.getLogger(__name__)


def random_subsample(X, target_nobs, replace=True):
    """ Draws subsets of cells uniformly at random. """

    nobs = X.shape[0]
    if (not replace) and (nobs <= target_nobs):
        return X
    else:
        indices = np.random.choice(nobs, size=target_nobs, replace=replace)
        return X[indices, :]


def weighted_subsample(X, w, target_nobs, replace=True, return_idx=False):
    nobs = X.shape[0]
    if (not replace) and (nobs <= target_nobs):
        return X
    else:
        indices = weighted_choice(w, target_nobs)
        if return_idx:
            return X[indices], indices
        else:
            return X[indices]


def weighted_choice(weights, nsample):
    rnd = np.random.random_sample(nsample) * sum(weights)
    selected_indices = np.empty(nsample, dtype=int)
    for i_sample, val in enumerate(rnd):
        accum = val
        iw = -1
        while accum >= 0:
            iw += 1
            accum -= weights[iw]
        selected_indices[i_sample] = iw
    return selected_indices


def kmeans_subsample(X, n_clusters, random_state=None, n_local_trials=10):
    """ Draws subsets of cells according to kmeans++ initialization strategy.
        Code slightly modified from sklearn, kmeans++ initialization. """

    random_state = check_random_state(random_state)
    n_samples, n_features = X.shape
    x_squared_norms = row_norms(X, squared=True)
    centers = np.empty((n_clusters, n_features))
    # Pick first center randomly
    center_id = random_state.randint(n_samples)
    centers[0] = X[center_id]
    # Initialize list of closest distances and calculate current potential
    closest_dist_sq = euclidean_distances(
        centers[0].reshape(1, -1), X, Y_norm_squared=x_squared_norms, squared=True)
    current_pot = closest_dist_sq.sum()

    # Pick the remaining n_clusters-1 points
    for c in range(1, n_clusters):
        # Choose center candidates by sampling with probability proportional
        # to the squared distance to the closest existing center
        rand_vals = random_state.random_sample(n_local_trials) * current_pot
        candidate_ids = np.searchsorted(closest_dist_sq.cumsum(), rand_vals)
        # Compute distances to center candidates
        distance_to_candidates = euclidean_distances(
            X[candidate_ids], X, Y_norm_squared=x_squared_norms, squared=True)
        # Decide which candidate is the best
        best_candidate = None
        best_pot = None
        best_dist_sq = None
        for trial in range(n_local_trials):
            # Compute potential when including center candidate
            new_dist_sq = np.minimum(closest_dist_sq,
                                     distance_to_candidates[trial])
            new_pot = new_dist_sq.sum()

            # Store result if it is the best local trial so far
            if (best_candidate is None) or (new_pot < best_pot):
                best_candidate = candidate_ids[trial]
                best_pot = new_pot
                best_dist_sq = new_dist_sq

        # Permanently add best center candidate found in local tries
        centers[c] = X[best_candidate]
        current_pot = best_pot
        closest_dist_sq = best_dist_sq

    return centers


def outlier_subsample(X, x_ctrl, to_keep, return_idx=False):
    """ Performs outlier selection. """

    outlier_scores = knn_dist(X, x_ctrl, s=100, p=1)
    indices = np.argsort(outlier_scores)[-to_keep:]
    if return_idx:
        return X[indices], outlier_scores[indices], indices
    else:
        return X[indices], outlier_scores[indices]


def knn_dist(x, x_ctrl, s=100, p=1):
    x_tmp = random_subsample(x_ctrl, 200000, replace=False)
    xs = kmeans_subsample(x_tmp, s)
    if p == 1:
        min_dist = np.min(pairwise_distances(X=x, Y=xs, metric='l1'), axis=1)
    elif p == 2:
        min_dist = np.min(pairwise_distances(X=x, Y=xs, metric='l2'), axis=1)
    assert len(min_dist) == x.shape[0]
    return min_dist


def knn_dist_memory_optimized(test_data, train_data, s):
    train_data = random_subsample(train_data, s, replace=False)
    nobs_test = test_data.shape[0]
    bs = 500
    test_kNN_dist = np.zeros(nobs_test)

    logger.info(f"Going up to: {nobs_test / bs + 1}")
    for ii in range(nobs_test / bs + 1):
        # is this a full batch or is it the last one?
        if (ii + 1) * bs < nobs_test:
            end = (ii + 1) * bs
        else:
            end = -1

        S = test_data[ii * bs:end]
        dist = pairwise_distances(X=S, Y=train_data, metric='l1')
        test_kNN_dist[ii * bs:end] = np.min(dist, axis=1)
    return test_kNN_dist
