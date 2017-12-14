from statistics import DPMixtureModel
import numpy
import pylab

if __name__ == '__main__':
    mus = numpy.array([[0,0],[0,.01],[.1,0],
                       [10,10],[10,10.1],[10.1,10]])
    xs = numpy.zeros((600,2))
    sigma = numpy.eye(2)
    xs[0:100,:] = numpy.random.multivariate_normal(mus[0,:], sigma, 100)
    xs[100:200,:] = numpy.random.multivariate_normal(mus[1,:], sigma, 100)
    xs[200:300,:] = numpy.random.multivariate_normal(mus[2,:], sigma, 100)
    xs[300:400,:] = numpy.random.multivariate_normal(mus[3,:], sigma, 100)
    xs[400:500,:] = numpy.random.multivariate_normal(mus[4,:], sigma, 100)
    xs[500:600,:] = numpy.random.multivariate_normal(mus[5,:], sigma, 100)
    true_z = numpy.zeros((600,))
    true_z[0:100] = 0
    true_z[100:200] = 1
    true_z[200:300] = 2
    true_z[300:400] = 3
    true_z[400:500] = 4
    true_z[500:600] = 5
    pylab.subplot(1,3,1)
    pylab.scatter(xs[:,0], xs[:,1], c=true_z)
    pylab.title("True")
    #pylab.show()
    model = DPMixtureModel(xs, 6, 5000, 100, 1)
    model.gamma = 1.0
    model.fit(verbose=True)
    pylab.subplot(1,3,2)
    model_z = model.get_class()
    results = model.get_results()
    model_class = results.classify(xs)
    pylab.scatter(xs[:,0],xs[:,1], c=model_class)
    pylab.scatter(results.mus()[:,0], results.mus()[:,1], c='red')
    pylab.title("component")
    modal = results.make_modal()
    z = modal.classify(xs)
    print z.min(), z.max()
    pylab.subplot(1,3,3)
    pylab.scatter(xs[:,0], xs[:,1], c=z)
    pylab.title("modal")
    pylab.show()
    ctot = 0 
    mtot = 0
    print "Assignment, component, modal"
    for i in range(8):
        ctot += len(numpy.where(model_z == i)[0])
        mtot += len(numpy.where(z == i)[0])
        print '%d: ' % i,len(numpy.where(model_class == i)[0]), len(numpy.where(z == i)[0])
    print "cmap: ",modal.cmap
    print ctot, mtot
    #modal.cmap = {0 : [0,1,2,3], 1 : [4,5,6,7]}
    print modal.prob(xs[0:10,:])
    print results.prob(xs[0:10,:])
    print modal.cmap

    print modal.prob(xs[0:10,:]).shape
    print results.prob(xs[0:10,:]).shape