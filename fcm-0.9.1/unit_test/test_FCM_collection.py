import unittest
from numpy import array, all, equal, sum, log10, where, all, isreal, eye
from fcm.core.transforms import _log_transform as log
from random import randint

from fcm import FCMdata
from fcm import FCMcollection
from numpy.ma.testutils import assert_array_equal, assert_equal
from fcm import PolyGate

class FCMcollectionTestCase(unittest.TestCase):
    def setUp(self):
        pnts = array([[1,1,1],[5,5,5]])
        fcm1 = FCMdata('test_fcm1', pnts, ['fsc','ssc','cd3'], [0,1])
        fcm2 = FCMdata('test_fcm2', pnts, ['fsc','ssc','cd3'], [0,1])
        self.fcms = FCMcollection('fcms', [fcm1, fcm2])
        
    def testCheckNames(self):
        pnts = array([[1,1,1],[5,5,5]])
        fcm1 = FCMdata('test_fcm1', pnts, ['fsc','ssc','cd3'], [0,1])
        fcm2 = FCMdata('test_fcm2', pnts, ['fsc','ssc','cd3'], [0,1])
        fcm3 = FCMdata('test_fcm3', pnts, ['fsc','ssc','cd4'], [0,1])

        fcms1 = FCMcollection('fcms1', [fcm1, fcm2])
        fcms2 = FCMcollection('fcms2', [fcm1, fcm2, fcm3, fcms1])
        
        check1 = fcms1.check_names()
        assert check1[fcms1.name] == [True, True, True]
        check2 = fcms2.check_names()
        assert check2[fcms2.name] == [check1, True, True, False]
        
    def testCompensate(self):
        # no fcmdata.compensate method???
        pass
        
    
    def testTransform(self):
        tmp = self.fcms['test_fcm1'][:]
        self.fcms.log([0,1,2])
        ltmp = array([[-1,-1,-1],[-1,-1,-1]])
        for i in range(3):
            ltmp[:,i] = where(tmp[:,i] <= 1, 0, log10(tmp[:,i]))

        assert_array_equal(self.fcms['test_fcm1'],ltmp , 'Log transform failed')
        assert_array_equal(self.fcms['test_fcm2'],ltmp , 'Log transform failed')
        
        self.fcms.logicle()
        # figure out what to assert...
    
    def testGate(self):
        verts =  array([[-.1,-.1],[-.1,1.1],[1.1,1.1], [1.1,-.1]])
        cols = [0,1]
        g = PolyGate(verts, cols)
        self.fcms.gate(g)
        assert_array_equal(self.fcms['test_fcm1'].view(), array([[1,1,1]]), 'Gating failed')
        assert_array_equal(self.fcms['test_fcm2'].view(), array([[1,1,1]]), 'Gating failed')
    
    def testFit(self):
        from fcm.statistics import DPMixtureModel, DPMixture
        k = 16
        niter = 10
        model = DPMixtureModel(k,niter,0,1)
        results = self.fcms.fit(model)
        assert_equal(len(results.keys()), 2, 'fitting produced wrong shaped dict') 
        for i in self.fcms:
            
            assert isinstance(results[i], DPMixture), 'fitting produced a non-DPMixture object'
        
    
    def testClassify(self):
        from fcm.statistics import DPCluster, DPMixture
        mu1 = array([0,0,0])
        sig = eye(3)
        mu2 = array([5,5,5])
        
        clst1 = DPCluster(.5, mu1, sig)
        clst2 = DPCluster(.5, mu2, sig)
        mix = DPMixture([clst1, clst2])
        
        cls = self.fcms.classify(mix)
        assert_array_equal(cls['test_fcm1'], array([0,1]), 'Calssify failed')
        assert_array_equal(cls['test_fcm2'], array([0,1]), 'Calssify failed')
    
    def testSummary(self):
        msg = self.fcms.summary()
        assert isinstance(msg,str), "summary failed"
        
if __name__ == '__main__':
    suite1 = unittest.makeSuite(FCMcollectionTestCase,'test')

    unittest.main()
