import unittest
from fcm import FCSreader
from fcm import loadFCS

class FCSreaderTestCase(unittest.TestCase):
    def setUp(self):
        self.fcm = FCSreader('../sample_data/3FITC_4PE_004.fcs').get_FCMdata()
        
    def testGetPnts(self):
        self.assertEqual(self.fcm.shape[0], int(self.fcm.notes.text['tot']))

    def testGetNotes(self):
        self.assertEqual(self.fcm.notes.text['cyt'], 'FACScan')
        
    def testMultiLoad(self):
        for unused in range(100):
            unused_x = loadFCS('../sample_data/3FITC_4PE_004.fcs', transform=None)
        

if __name__ == '__main__':
    suite1 = unittest.makeSuite(FCSreaderTestCase,'test')

    unittest.main()
