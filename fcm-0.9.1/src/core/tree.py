import re
import numpy as np


class Node(object):
    """
    base node object
    """

    def __init__(self, name, parent, data):
        self.name = name
        self.parent = parent
        self.data = data
        self.prefix = 'n'

    def view(self):
        """
        return the view of the data associated with this node
        """

        return self.data

    def pprint(self, depth, size):
        tmp = "  " * depth + self.name
        if size:
            tmp = tmp + " " + str(self.view().shape[0])
    
        return tmp + "\n"

    def __getattr__(self, name):
        if name == 'channels':
            return self.parent.channels
        else:
            raise AttributeError("'%s' has no attribue '%s'" % (str(self.__class__), name))

class RootNode(Node):
    """
    Root Node
    """
    def __init__(self, name, data, channels):
        self.name = name
        self.parent = None
        self.data = data
        self.channels = channels
        self.prefix = 'root'

class TransformNode(Node):
    """
    Transformed Data Node
    """

    def __init__(self, name, parent, data):
        self.name = name
        self.parent = parent
        self.data = data
        self.prefix = 't'

    def __getattr__(self, name):
        if name == 'channels':
            return self.parent.channels
        else:
            raise AttributeError("'%s' has no attribue '%s'" % (str(self.__class__), name))
        
class CompensationNode(TransformNode):
    '''
    Compensated Data Node
    '''

    def __init__(self, name, parent, data, sidx, spill):
        self.name = name
        self.parent = parent
        self.data = data
        self.sidx = sidx
        self.spill = spill
        self.prefix = 'c'
        

class SubsampleNode(Node):
    """
    Node of subsampled data
    """

    def __init__(self, name, parent, param):
        self.name = name
        self.parent = parent
        self.param = param
        self.prefix = 's'
        if isinstance(param,tuple):
            self.channels = self.parent.channels[param[1]]

    def view(self):
        """
        return the view of the data associated with this node
        """
        return self.parent.view().__getitem__(self.param)

class DropChannelNode(Node):
    """
    Node of data removing specific channels
    """

    def __init__(self, name, parent, param, channels):
        self.name = name
        self.parent = parent
        self.param = param
        self.prefix = 'd'
        self.channels = channels

    def view(self):
        """
        return the view of the data associated with this node
        """
        return self.parent.view()[:, self.param]


class GatingNode(Node):
    """
    Node of gated data
    """

    def __init__(self, name, parent, data):
        self.name = name
        self.parent = parent
        self.data = data
        self.prefix = 'g'

    def view(self):
        """
        return the view of the data associated with this node
        """
        if self.parent.view().shape[0] == 0:
            return np.array([]).reshape(self.parent.view().shape)
        return self.parent.view()[self.data]

    def __getattr__(self, name):
        if name == 'channels':
            return self.parent.channels
        else:
            raise AttributeError("'%s' has no attribue '%s'" % (str(self.__class__), name))

class Tree(object):
    '''Tree of data for FCMdata object.'''

    def __init__(self, pnts, channels):
        self.nodes = {}
        self.root = RootNode('root', pnts, channels)
        self.nodes['root'] = self.root
        self.current = self.root

    def parent(self):
        '''return the parent of a node'''
        return self.current.parent

    def children(self , node=None):
        '''return the children of a node'''
        if node == None:
            node = self.current
        return [i for i in self.nodes.values() if i.parent == node]

    def visit(self, name):
        '''visit a node in the tree'''
        if isinstance(name, str):
            self.current = self.nodes[name]
        elif isinstance(name, Node): # in this case we assume we're a node type.
            self.current = name
        else:
            raise KeyError("No such Node %s" % str(name))

    def get(self, name=None):
        '''return the current node object'''
        if name is None:
            return self.current
        else:
            if name in self.nodes:
                return self.nodes[name]
            else:
                raise KeyError, 'No node named %s' % name

    def view(self):
        '''Return a view of the current data'''
        return self.current.view()

    def add_child(self, name, node):
        '''Add a node to the tree at the currently selected node'''
        if name == '':
            prefix = node.prefix
            pat = re.compile(prefix + "(\d+)")
            matches = [pat.search(i) for i in self.nodes]
            matches = [i for i in matches if i is not None]
            if len(matches): # len > 0
                n = max([ int(i.group(1)) for i in matches])
                name = prefix + str(n + 1)
            else:
                name = prefix + '1'
        if name in self.nodes.keys():
            raise KeyError, 'name, %s, already in use in tree' % name
        else:
            node.name = name
            self.nodes[name] = node
            node.parent = self.current
            self.current = self.nodes[name]

    def rename_node(self, old_name, new_name):
        """
        Rename a node name
        D(old,new) -> rename old to new
        """
        if not self.nodes.has_key(old_name):
            # we don't have old_name...
            raise KeyError, 'No node named %s' % old_name
        if self.nodes.has_key(new_name):
            raise KeyError, 'There already is a node name %s' % new_name
        else:
            self.nodes[new_name] = self.nodes[old_name] # move node
            self.nodes[new_name].name = new_name # fix it's name
            del self.nodes[old_name] # remove old node.


    def pprint(self, size=False):
        return self._rpprint(self.root, 0, size)

    def _rpprint(self, n, d, size=False):
        tmp = n.pprint(d, size)
        for i in self.children(n):
            tmp += self._rpprint(i, d + 1, size)
        return tmp

if __name__ == '__main__':
    pass


