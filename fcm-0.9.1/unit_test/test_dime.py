"""Test dime code for simple cases."""

from __future__ import division
import numpy
import pylab
from fcm.statistics import dime

if __name__ == '__main__':
    mu0 = [0,0,0]
    mu1 = [1,0,0]
    mu2 = [2,2.5,0]
    mu3 = [3,2.5,0]

    p = len(mu0)

    sigma = numpy.identity(3)
#    sigma = numpy.array([[1,0,.5],
#                         [0,1,0],
#                         [.5,0,1]])
    x0 = numpy.random.multivariate_normal(mu0, sigma, 50)
    x1 = numpy.random.multivariate_normal(mu1, sigma, 50)
    x2 = numpy.random.multivariate_normal(mu2, sigma, 1250)
    x3 = numpy.random.multivariate_normal(mu3, sigma, 1250)

    x = numpy.concatenate([x0, x1, x2, x3])

    adict = {
        0 : [0,1],
        1 : [2,3]
        }

    pis = numpy.array([len(x0)/len(x), len(x1)/len(x), len(x2)/len(x), len(x3)/len(x)])
    mus = numpy.array([mu0, mu1, mu2, mu3])
    sigmas = numpy.array([sigma, sigma, sigma, sigma])

    #info = dime.DiME(x, pis, mus, sigmas, cmap=adict)
    info = dime.Dime(pi=pis, mu=mus, sigma = sigmas, cmap=adict)
    
    const = info.drop(0,[0,1,2])
    infos = [(info.drop(0,i)-const)/(1-const) for i in range(p)]
    infos2 = [(info.drop(0,[0,1])-const)/(1-const), (info.drop(0,[0,2])-const)/(1-const), (info.drop(0,[1,2])-const)/(1-const)]

    print const
    print infos
    print infos2
    
    pylab.scatter(x0[:,0], x0[:,1], c='r')
    pylab.scatter(x1[:,0], x1[:,1], c='r')
    pylab.scatter(x2[:,0], x2[:,1], c='b')
    pylab.scatter(x3[:,0], x3[:,1], c='b')
    pylab.show()
