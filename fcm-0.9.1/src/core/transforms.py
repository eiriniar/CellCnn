"""Base functions for various transforms to be used on FCM data
"""

from scipy.optimize import brentq
from scipy import interpolate
from numpy import array, abs, arange, exp, log, log10, min, max, sign, concatenate, vectorize, where

from tree import TransformNode

import logicle as clogicle

def quantile(x, n):
    """return the lower nth quantile"""
    try:
        return sorted(x)[int(n * len(x))]
    except IndexError:
        return 0


def productlog(x, prec=1e-12):
    """Productlog or LambertW function computes principal solution for w in f(w)
 = w*exp(w)."""
    #  fast estimate with closed-form approximation
    if (x <= 500):
        lxl = log(x + 1.0)
        return 0.665 * (1 + 0.0195 * lxl) * lxl + 0.04
    else:
        return log(x - 4.0) - (1.0 - 1.0 / log(x)) * log(log(x))

def S(x, y, T, m, w):
    p = w / (2 * productlog(0.5 * exp(-w / 2) * w))
    sgn = sign(x - w)
    xw = sgn * (x - w)
    return sgn * T * exp(-(m - w)) * (exp(xw) - p ** 2 * exp(-xw / p) + p ** 2 - 1) - y

def _logicle(y, T, m, r, w):
    y = array(y, dtype='double')
    if w is None:
        if r == 0:
            w = 1 # don't like this but it works... FIX!
        else:
            w = (m - log10(T / abs(r))) / 2.0

    clogicle.logicle_scale(T, w, m, 0, y)
    return y

def logicle(fcm, channels, T, m, r=None, scale_max=1e5, scale_min=0, w=None, rquant=None):
    """return logicle transformed points in fcm data for channels listed"""
    npnts = fcm.view().copy()
    for i in channels:
        if rquant is not None:
            w = None
            tmp = npnts[:,i]
            r = quantile(tmp[tmp<0], 0.05)
        if r is None and w is None:
            w = .5
        tmp = scale_max * _logicle(npnts[:, i].T, T, m, r, w)
        #tmp[tmp<scale_min] = scale_min
        npnts.T[i] = tmp
    node = TransformNode('', fcm.get_cur_node(), npnts)
    fcm.add_view(node)
    return fcm

def EH(x, y, b, d, r):
    e = float(d) / r
    sgn = sign(x)
    return sgn * 10 ** (sgn * e * x) + b * e * x - sgn - y

def hyperlog0(y, b, d, r):
    return brentq(EH, -10 ** 6, 10 ** 6, (y, b, d, r))
hyperlog0 = vectorize(hyperlog0)

def _hyperlog(y, b, d, r, order=2, intervals=1000.0):
    ub = log(max(y) + 1 - min(y))
    xx = exp(arange(0, ub, ub / intervals)) - 1 + min(y)
    yy = hyperlog0(xx, b, d, r)
    t = interpolate.splrep(xx, yy, k=order)
    return interpolate.splev(y, t)

def hyperlog(fcm, channels, b, d, r, order=2, intervals=1000.0):
    npnts = fcm.view().copy()
    for i in channels:
        npnts.T[i] = _hyperlog(npnts[:, i].T, b, d, r, order=2, intervals=1000.0)
    node = TransformNode('', fcm.get_cur_node(), npnts)
    fcm.add_view(node)
    return fcm

def log_transform(fcm, channels):
    npnts = fcm.view().copy()
    for i in channels:
        #npnts[:,i] = where(npnts[:,i] <= 1, 0, log10(npnts[:,i]))
        npnts[:, i] = _log_transform(npnts[:, i])
    node = TransformNode('', fcm.get_cur_node(), npnts)
    fcm.add_view(node)
    return fcm

def _log_transform(npnts):
    return where(npnts <= 1, 0, log10(npnts))

if __name__ == '__main__':
    from numpy.random import normal, lognormal
    import numpy
    import pylab


    d1 = normal(0, 50, (50000))
    d2 = lognormal(8, 1, (50000))
    d3 = concatenate([d1, d2])

    T = 262144
    d = 4
    #m = d*log(10)
    m = 4.5
    print m
    r = quantile(d3[d3 < 0], 0.05)
    w = (m - log(T / abs(r))) / 2
    w = (m - numpy.log10(T / numpy.abs(r))) / 2.0
    #if (w<0):
    #w = .5
    print w
    print _logicle([0, T], T, m, r)
    n = array([0, T], dtype='double')
    clogicle.logicle_scale(T, w, 4.5, 0, n)
    print n
    lmin, lmax = _logicle([0, T], T, m, r)
    pylab.clf()
    pylab.figtext(0.5, 0.94, 'Logicle transform with r=%.2f, d=%d and T=%d\nData is normal(0, 50, 50000) + lognormal(8, 1, 50000)' % (r, d, T),
                  va='center', ha='center', fontsize=12)

    pylab.subplot(4, 1, 1)
    x = arange(0, m, 0.1)
    pylab.plot(x, S(x, 0, T, m, w))
    locs, labs = pylab.xticks()
    pylab.xticks([])
    pylab.yticks([])
    pylab.ylabel('Inverse logicle')

    pylab.subplot(4, 1, 2)
    pylab.hist(d3, 1250)
    locs, labs = pylab.xticks()
    pylab.xticks([])
    pylab.yticks([])
    pylab.ylabel('Raw data')

    pylab.subplot(4, 1, 3)
    d = 1e5 / lmax * _logicle(d3, T, m, r)
    d[d < 0] = 0
    pylab.hist(d, 1250)
    locs, labs = pylab.xticks()
    #pylab.xticks([])
    pylab.yticks([])
    pylab.ylabel('Data after transform')
    pylab.subplot(4, 1, 4)
    d = array(d3, dtype='double')
    tmp = array([0.0, T])
    clogicle.logicle_scale(T, w, m, 0, tmp)
    l_min = tmp[0]
    l_max = tmp[1]
    clogicle.logicle_scale(T, w, m, 0, d)
    #d[d<0]=0
    pylab.hist(1e5 / l_max * d, 1250)
    pylab.yticks([])
    #pylab.ylim((0,600))
    pylab.ylabel('Data after transform')
    # pylab.savefig('logicle.png')
    pylab.show()

