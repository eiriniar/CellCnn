import unittest
from fcm import Annotation

class FCMAnnotationTestCase(unittest.TestCase):
    def setUp(self):
        self.test = {'foo': 'bar'}
        self.ann = Annotation(self.test)
    
    def testFlatName(self):
        assert self.ann.foo == 'bar', 'flat name lookup failed'
        assert self.ann['foo'] == 'bar', 'index lookup failed'
        assert self.ann.foo == self.ann['foo'], 'flat lookup isnt index lookup'
    
    def testFlatAssign(self):
        self.ann.spam = 'eggs'
        assert self.ann['spam'] == 'eggs', 'assignment lookup by index failed'
        assert self.ann.spam == 'eggs', 'assignment lookup by flat failed'
        
    def testIndexAssign(self):
        self.ann['spam'] = 'eggs'
        assert self.ann['spam'] == 'eggs', 'assignment lookup by index failed'
        assert self.ann.spam == 'eggs', 'assignment lookup by flat failed'
    
    def testAnnDeleg(self):
        assert self.ann.keys()[0] == self.test.keys()[0], 'delegation of keys() failed'
        
if __name__ == '__main__':
    suite1 = unittest.makeSuite(FCMAnnotationTestCase,'test')

    unittest.main()
