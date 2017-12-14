import unittest
from fcm.statistics import DPCluster
from numpy import array, eye, dot
import numpy.testing
class Dp_clusterTestCase(unittest.TestCase):


    def setUp(self):
        self.mu1 = array([0,0,0])
        self.sig = eye(3)
        self.mu2 = array([5,5,5])
        
        self.clst1 = DPCluster(.5, self.mu1, self.sig)
        self.clst2 = DPCluster(.5, self.mu2, self.sig)
        
    def testadd(self):
        
        b = self.clst1 + 2

        self.assertIsInstance(b, DPCluster, 'addition returned wrong type')
        
        numpy.testing.assert_array_equal(b.mu, self.mu1+2, 
                                         "integer addition to cluster failed")
        numpy.testing.assert_array_equal(b.mu, self.clst1.mu+2, 
                                         "integer addition to cluster failed")
        b = 2 + self.clst2
        
        self.assertIsInstance(b, DPCluster, 'addition returned wrong type')
        numpy.testing.assert_array_equal(b.mu, self.mu2+2, 
                                         "integer addition to cluster failed")
        numpy.testing.assert_array_equal(b.mu, self.clst2.mu+2, 
                                         "integer addition to cluster failed")
        
        b = self.clst1 + 2.5

        self.assertIsInstance(b, DPCluster, 'addition returned wrong type')
        numpy.testing.assert_array_equal(b.mu, self.mu1+2.5, 
                                         "float addition to cluster failed")
        numpy.testing.assert_array_equal(b.mu, self.clst1.mu+2.5, 
                                         "float addition to cluster failed")
        b = 2.5 + self.clst2
        
        self.assertIsInstance(b, DPCluster, 'addition returned wrong type')
        numpy.testing.assert_array_equal(b.mu, self.mu2+2.5, 
                                         "float addition to cluster failed")
        numpy.testing.assert_array_equal(b.mu, self.clst2.mu+2.5, 
                                         "float addition to cluster failed")
        
        adder = array([0,1,2])
        b = self.clst1 + adder
        self.assertIsInstance(b, DPCluster, 'addition returned wrong type')
        numpy.testing.assert_array_equal(b.mu, self.mu1+adder, 
                                         "array addition to cluster failed")
        numpy.testing.assert_array_equal(b.mu, self.clst1.mu+adder, 
                                         "array addition to cluster failed")
        adder = array([3,4,5])
        b = adder + self.clst2
        self.assertIsInstance(b, DPCluster, 'addition returned wrong type')
        numpy.testing.assert_array_equal(b.mu, self.mu2+adder, 
                                         "array addition to cluster failed")
        numpy.testing.assert_array_equal(b.mu, self.clst2.mu+adder, 
                                         "array addition to cluster failed")
        
    def testsub(self):
        b = self.clst1 - 2
        self.assertIsInstance(b, DPCluster, 'subtraction returned wrong type')
        numpy.testing.assert_array_equal(b.mu, self.mu1-2, 
                                         "integer subtraction to cluster failed 1")
        numpy.testing.assert_array_equal(b.mu, self.clst1.mu-2, 
                                         "integer subtraction to cluster failed 2")
        b = 2 - self.clst2
        self.assertIsInstance(b, DPCluster, 'subtraction returned wrong type')
        numpy.testing.assert_array_equal(b.mu, 2 - self.mu2, 
                                         "integer subtraction to cluster failed 3")
        numpy.testing.assert_array_equal(b.mu, 2 - self.clst2.mu, 
                                         "integer subtraction to cluster failed 4")
        
        adder = array([3,4,5])
        b =  self.clst1 - adder
        
        self.assertIsInstance(b, DPCluster, 'subtraction returned wrong type')
        numpy.testing.assert_array_equal(b.mu, self.mu1-adder, 
                                         "array subtraction to cluster failed 1")
        numpy.testing.assert_array_equal(b.mu, self.clst1.mu-adder, 
                                         "array subtraction to cluster failed 2")
        
        adder = array([3,4,5])
        b = adder - self.clst2
        
        self.assertIsInstance(b, DPCluster, 'subtraction returned wrong type')
        numpy.testing.assert_array_equal(b.mu, adder - self.mu2, 
                                         "array subtraction to cluster failed 1")
        numpy.testing.assert_array_equal(b.mu, adder - self.clst2.mu, 
                                         "array subtraction to cluster failed 2")
        
    def testmul(self):
        b = self.clst1 * 2
        
        self.assertIsInstance(b, DPCluster, 'subtraction returned wrong type')
        numpy.testing.assert_array_equal(b.mu, self.mu1 * 2, 
                                         "int mult to cluster failed 1")
        numpy.testing.assert_array_equal(b.mu, self.clst1.mu * 2, 
                                         "int mult to cluster failed 2")
        
        numpy.testing.assert_array_equal(b.sigma, self.clst1.sigma * 2 * 2,
                                         "int mult to cluster failed 2")
        
        b = 2*self.clst1
        
        self.assertIsInstance(b, DPCluster, 'subtraction returned wrong type')
        numpy.testing.assert_array_equal(b.mu, self.mu1 * 2, 
                                         "int mult to cluster failed 1")
        numpy.testing.assert_array_equal(b.mu, self.clst1.mu * 2, 
                                         "int mult to cluster failed 2")
        
        numpy.testing.assert_array_equal(b.sigma, self.clst1.sigma * 2 * 2,
                                         "int mult to cluster failed 2")
        
        
        adder = array([2,3,4])
        b = adder*self.clst1
        
        self.assertIsInstance(b, DPCluster, 'subtraction returned wrong type')
        numpy.testing.assert_array_equal(b.mu, dot(adder, self.mu1), 
                                         "array mult to cluster failed 1")
        numpy.testing.assert_array_equal(b.mu, dot(adder, self.clst1.mu), 
                                         "array mult to cluster failed 2")
        numpy.testing.assert_array_equal(b.sigma, 
                                         dot(dot(adder, self.sig),adder.T),
                                         'array mult to cluster failed')