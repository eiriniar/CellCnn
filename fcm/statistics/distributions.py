"""
Distributions used in FCS analysis
"""

from numpy import array, sum, cumsum, reshape, exp, ones, ndarray
from numpy.random import random, multivariate_normal


try:
    from gpustats import mvnpdf_multi
    #from gpustats.util import threadSafeInit
    from dpmix.utils import select_gpu
    has_gpu = True
except ImportError:
    has_gpu = False

from dpmix.utils import mvn_weighted_logged
#def mvnormpdf(x, mu, va):
#    """
#    multi variate normal pdf, derived from David Cournapeau's em package
#    for mixture models
#    http://www.ar.media.kyoto-u.ac.jp/members/david/softwares/em/index.html
#    """
#    d       = mu.size
#    inva    = inv(va)
#    fac     = 1 /sqrt( (2*pi) ** d * fabs(det(va)))
#
#    y   = -0.5 * dot(dot((x-mu), inva) * (x-mu), 
#                       ones((mu.size, 1), x.dtype))
#
#    y   = fac * exp(y)
#    return y

def _mvnpdf(x, mu, va, n=1, logged=False, use_gpu=True, **kwargs):
    if len(x.shape) == 1:
            x = x.reshape((1,x.shape))
    if len(mu.shape) == 1:
        mu = mu.reshape((1,mu.shape))
    if len(va.shape) == 2:
        va = va.reshape(1,va.shape[0], va.shape[1])
        
    if has_gpu and use_gpu:
        if 'device' in kwargs:
            dev = kwargs['device']
        else:
            dev = 0
        select_gpu(dev)
        return mvnpdf_multi(x, mu, va, weights=ones(mu.shape[0]), logged=logged, order='C').astype('float64')
    else:
        if logged:
            return mvn_weighted_logged(x, mu, va, ones(mu.shape[0]))
        else:
            return exp(mvn_weighted_logged(x, mu, va, ones(mu.shape[0])))

def _wmvnpdf(x, pi, mu, va, n=1, logged=False, use_gpu=True, **kwargs):
    if len(x.shape) == 1:
            x = x.reshape((1,x.shape))
    if len(mu.shape) == 1:
        mu = mu.reshape((1,mu.shape))
    if len(va.shape) == 2:
        va = va.reshape(1,va.shape[0], va.shape[1])
    if isinstance(pi, float) or isinstance(pi, int):
        pi = array([pi])
    elif isinstance(pi, ndarray):
        if len(pi.shape) == 0:
            pi = pi.reshape((1))
    
    if has_gpu and use_gpu:
        if 'device' in kwargs:
            dev = kwargs['device']
        else:
            dev = 0
        select_gpu(dev)
        return mvnpdf_multi(x, mu, va, weights = pi, logged=logged, order='C').astype('float64')
    else:
        if logged:
            return mvn_weighted_logged(x, mu, va, pi)
        else:
            return exp(mvn_weighted_logged(x, mu, va, pi))
    

def mvnormpdf(x, mu, va, **kwargs):
    """
    calculate the multi-variate normal pdf 
    D(x, mu, sigma) -> float
    """
    try:
        n, p = x.shape
    except ValueError:
        n = x.shape
        p = 0
    if p > 0:
        results = _mvnpdf(x, mu, va, n, **kwargs)
    else:
        results = _mvnpdf(x, mu, va, 1, **kwargs)

    return results

def compmixnormpdf(x, prop, mu, Sigma, **kwargs):
    """Component mixture multivariate normal pdfs"""
    try:
        n, d = x.shape
    except ValueError:
        d = x.shape[0]
        n = 1
        x = x.reshape((1, d))
    try:
        c = prop.shape[0]
    except AttributeError:
        c = 1
    except IndexError:
        c = 1

    if c == 1:
        tmp = _wmvnpdf(x, prop, mu, Sigma, n, **kwargs)
        if n == 1:
            tmp = tmp[0]

    else:
        tmp = _wmvnpdf(x, prop, mu, Sigma, n * c, **kwargs)
        tmp = reshape(tmp, (n, c))
        #tmp = sum(tmp,1)
        if n == 1:
            tmp = tmp[0]
    return tmp

def mixnormpdf(x, prop, mu, Sigma, **kwargs):
    """Mixture of multivariate normal pdfs"""
    # print "in mixnormpdf ..."
#    tmp = 0.0
#    for i in range(len(prop)):
#        tmp += prop[i]*mvnormpdf(x, mu[i], Sigma[i])
#    return tmp
    tmp = compmixnormpdf(x, prop, mu, Sigma, **kwargs)
    try:
        return sum(tmp, 1)
    except ValueError:
        return sum(tmp, 0)


def mixnormrnd(pi, mu, sigma, k):
    """Generate random variables from mixture of Guassians"""
    xs = []
    for unused in range(k):
        j = sum(random() > cumsum(pi))
        xs.append(multivariate_normal(mu[j], sigma[j]))
    return array(xs)

if __name__ == '__main__':
    x = array([1, 0])
    mu = array([5, 5])
    sig = array([[1, 0], [0, 1]])

    print 'new:', mvnormpdf(x, mu, sig)
    x = array([1, 0])
    mu = array([0, 0])
    sig = array([[1, .75], [.75, 1]])

    print 'new:', mvnormpdf(x, mu, sig)
    print 'array:', mvnormpdf(array([x, x - 2]), mu, sig)

    x = array([[1, 0], [5, 5], [0, 0]])
    mu = array([[0, 0], [5, 5]])
    sig = array([[[1, .75], [.75, 1]], [[1, 0], [0, 1]]])
    p = array([.5, .5])
    print 'mix:', mixnormpdf(x, p, mu, sig)
    #print 'mix:', mixnormpdf(x[0],p,mu,sig)

