import unittest
import numpy
from fcm.core import productlog

class FCMtransformTestCase(unittest.TestCase):
    def setUp(self):
        pass
        
    def testProductLog(self):
        ans = numpy.array([0., 0.567143, 1.74553, 3.38563, 5.2496, 7.23185])
        for i,x in enumerate([0,1,10,100,1000,10000]):
            self.assert_(numpy.abs(productlog(x) - ans[i]) < 0.1)

if __name__ == '__main__':
    suite1 = unittest.makeSuite(FCMtransformTestCase, 'test')

    unittest.main()
