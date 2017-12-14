'''
Created on Nov 5, 2009

@author: Jacob Frelinger
'''
#from scipy.optimize import fmin
#from np import around, array, log10   
from __future__ import division
from numpy.linalg import solve, inv
from collections import deque
from itertools import chain
import numpy as np
import numpy.random as npr
from scipy.spatial.distance import pdist as _pdist, squareform

from fcm.statistics.distributions import compmixnormpdf, mixnormpdf, mvnormpdf, mixnormrnd

def bfs(g, start):
    """BFS generator from start node"""
    queue, enqueued = deque([(None, start)]), set([start])
    while queue:
        parent, n = queue.popleft()
        yield parent, n
        new = set(g[n]) - enqueued
        enqueued |= new
        queue.extend([(n, child) for child in new])

def dfs(g, start):
    """DFS generator from start node"""
    stack, enqueued = [(None, start)], set([start])
    while stack:
        parent, n = stack.pop()
        yield parent, n
        new = set(g[n]) - enqueued
        enqueued |= new
        stack.extend([(n, child) for child in new])

def flatten(listOfLists):
    "Flatten one level of nesting"
    return chain.from_iterable(listOfLists)

def nodes(collection):
    """Return unique nodes"""
    return tuple(set(node for node in flatten(collection) if node is not None))

def find_components(g):
    """Find connected components in graph g"""
    return np.unique([nodes(bfs(g, i)) for i in range(len(g))])

def matrix_to_graph(m):
    """Convert adjacency matrix to dictionary form of graph"""
    graph = {}
    for i, row in enumerate(m):
        graph[i] = np.nonzero(m[i, :])[0]
    return graph

def pdist(x, w=None, scale=False):
    """Returns the distance matrix of points x. If scale is true,
    rescale distance by sqrt(number of dimensions).  If w is provided,
    it weights the original matrix of points *before* calculating the
    distance for efficiency rather than weighting of distances."""
    n, p = x.shape
    if w is not None:
        #w = w / np.sum(w.astype(np.float))
        w = np.sqrt(w)
        print w
        print x
        x = x * w
        print x
    if scale:
        return (1.0 / np.sqrt(p)) * squareform(_pdist(x, 'euclidean'))
    else:
        return squareform(_pdist(x, 'euclidean'))


def modesearch(pis, mus, sigmas, tol=1e-6, maxiter=20, delta=.1, w=None, scale=False):
    """find the modes of a mixture of guassians"""

    mdict, sm, unused_spm = _mode_search(pis, mus, sigmas, nk=0, tol=tol, maxiter=maxiter)

    m = np.array([i[0] for i in mdict.values()])
    sm = np.array(sm)
    xs = np.zeros((len(mdict.keys()), mus.shape[1]))
    # use stored index as dict items are not ordered
    for j, key in enumerate(mdict):
        cur_mode = tuple(m[j, :].tolist())
        xs[key[0], :] = cur_mode

    dm = pdist(xs, scale=scale, w=w) < delta#, w=np.array([1,1,1,1]))
    #print 'm', m
    cs = find_components(matrix_to_graph(dm))
    cs = sorted(cs, key=len, reverse=True)
    
    rslt = {}
    modes = {}
    for i, j in enumerate(cs):
        modes[i] = np.mean(np.vstack([xs[k, :] for k in j]), 0)
        rslt[i] = j

    return modes, rslt




def _mode_search(pi, mu, sigma, nk=0, tol=0.000001, maxiter=20):
    """Search for modes in mixture of Gaussians"""
    k, unused_p = mu.shape
    omega = np.copy(sigma)
    a = np.copy(mu)

    for j in range(k):
        omega[j] = inv(sigma[j])
        a[j] = solve(sigma[j], mu[j])

    if nk > 0:
        allx = np.concatenate([mu, mixnormrnd(pi, mu, sigma, nk)])
    else:
        allx = np.copy(mu)
    allpx = mixnormpdf(allx, pi, mu, sigma, use_gpu=False)
    nk += k

    mdict = {} # modes
    sm = [] # starting point of mode search
    spm = [] # density at starting points

    etol = np.exp(tol)
    # rnd = int(-1*np.floor(np.log10(tol)))
    rnd = 1

    for js in range(nk):
        x = allx[js]
        px = allpx[js]
        sm.append(x)
        spm.append(px)
        # w = compmixnormpdf(allx,pi,mu,sigma)
        h = 0
        eps = 1 + etol

        while ((h <= maxiter) and (eps > etol)):
            w = compmixnormpdf(x, pi, mu, sigma, use_gpu=False)
            Y = np.sum([w[j] * omega[j] for j in range(k)], 0)
            yy = np.dot(w, a)
            y = solve(Y, yy)
            py = mixnormpdf(y, pi, mu, sigma, use_gpu=False)
            eps = py / px
            x = y
            px = py
            h += 1

        mdict[(js, tuple(x))] = [x, px] # eliminate duplicates
    
    return mdict, sm, spm

def check_mode(m, pm, pi, mu, sigma):
    """Check that modes are local maxima"""
    k, p = mu.shape
    z = np.zeros(p)
    tm = [] # true modes
    tpm = [] # true mode densities
    for _m, _pm in zip(m, pm):
        G = np.zeros((p, p))
        omega = sigma.copy()
        for j in range(k):
            omega[j] = inv(sigma[j])
            eij = _m - mu[j]
            S = np.dot(np.identity(p) - np.outer(eij, eij),
                          omega[j])
            G += pi[j] * mvnormpdf(eij, z, sigma[j], use_gpu=False) * S
        if np.linalg.det(G) > 0:
            tm.append(_m)
            tpm.append(_pm)
    return np.array(tm), np.array(tpm)


if __name__ == '__main__':
    pi = np.array([0.3, 0.4, 0.29, 0.01])
    mu = np.array([[0, 0],
                      [1, 3],
                      [-3, -1],
                      [1.2, 2.8]], 'd')
    sigma = np.array([1 * np.identity(2),
                         1 * np.identity(2),
                         1 * np.identity(2),
                         1 * np.identity(2)])
    modes, rslt = modesearch(pi, mu, sigma)

    for key in modes:
        print key, modes[key]
    print
    for key in rslt:
        print key, rslt[key]
