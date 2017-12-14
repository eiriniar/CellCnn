import unittest
from fcm.core.tree import Tree, RootNode


class TreeTestCase(unittest.TestCase):
    def setUp(self):
        self.t = Tree(RootNode('root',[0,0,0], []), [])
        self.t.add_child('gate1', RootNode('gate1',[1,2,3], [] ))
        self.t.add_child('gate11', RootNode('gate11',[4,5,6], []))
        self.t.visit('root')
        self.t.add_child('gate2', RootNode('gate2',[2,3,4], []))
        self.t.add_child('gate21', RootNode('gate21',[3,4,5], []))
        self.t.add_child('gate211', RootNode('gate211',[4,5,6], []))
        
    def testView(self):
        assert self.t.view() == [4,5,6], 'view current node failed'
    
    def testRename(self):
        self.t.rename_node('gate1', 'foo')
        self.t.visit('foo')
        assert self.t.get().name == 'foo', 'rename failed to change node name'
        assert self.t.view() == [1,2,3], 'rename gate1 to foo failed'
        self.assertRaises(KeyError, self.t.rename_node, 'foo', 'gate2')
        
    def testVisitError(self):
        self.t.visit('root')
        self.assertRaises(KeyError, self.t.visit, 2)
        self.assertRaises(KeyError, self.t.visit, 'this node does not exist')
        
if __name__ == '__main__':
    suite1 = unittest.makeSuite(TreeTestCase,'test')

    unittest.main()