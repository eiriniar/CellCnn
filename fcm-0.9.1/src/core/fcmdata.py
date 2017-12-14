"""
A python object representing flow cytomoetry data
"""
from __future__ import division
from numpy import median, log
from annotation import Annotation
from transforms import logicle as _logicle
from transforms import hyperlog as _hyperlog
from transforms import log_transform as _log
from tree import Tree
from fcm.core.compensate import compensate
#from fcm.io.export_to_fcs import export_fcs


class FCMdata(object):
    """
    Object representing flow cytometry data
    FCMdata.pnts : a numpy array of data points
    FCMdata.channels : a list of which markers/scatters are on which column of
    the array.
    FCMdata.scatters : a list of which indexes in fcmdata.channels are scatters

    """

    def __init__(self, name, pnts, channels, scatters=None, notes=None):
        """
        fcmdata(name, pnts, channels, scatters=None)
        name: name of corresponding FCS file minus extension
        pnts: array of data points
        channels: a list of which markers/scatters are on which column of
                    the array.
        scatters: a list of which indexes in channels are scatters
        
        """
        self.name = name
#        if type(pnts) != type(array([])):
#            raise BadFCMPointDataTypeError(pnts, "pnts isn't a numpy.array")
        self.tree = Tree(pnts, channels)
        #self.pnts = pnts
        #self.channels = channels
        #TODO add some default intelegence for determining scatters if None
        self.scatters = scatters
        self.markers = []
        if self.scatters is not None:
            for chan in range(len(channels)):
                if chan in self.scatters:
                    pass
                elif self.tree.root.channels[chan] in self.scatters:
                    pass
                else:
                    self.markers.append(chan)
        if notes == None:
            notes = Annotation()
        self.notes = notes

    def __unicode__(self):
        return self.name

    def __repr__(self):
        return self.name

    def __getitem__(self, item):
        """return FCMdata points"""
        
        if isinstance(item,tuple):
            
            item = list(item) # convert to be mutable.
            if isinstance(item[1],str):
                
                item[1] = self.name_to_index(item[1])
            elif isinstance(item[1],tuple) or isinstance(item[1], list):
                
                item[1] = list(item[1])# convert to be mutable.
                for i,j in enumerate(item[1]):
                    if isinstance(j,str):
                        print i, 'is string', j
                        item[1][i] = self.name_to_index(j)
            item = tuple(item)
                        
        
        return self.tree.view()[item]

    @property
    def channels(self):
        return self.current_node.channels
    
    def __getattr__(self, name):
            if name in dir(self.current_node.view()):
                #return Node.__getattribute__(self.current_node,'view')().__getattribute__(name)
                return self.current_node.view().__getattribute__(name)
            else:
                raise AttributeError("'%s' has no attribue '%s'" % (str(self.__class__), name))

    def __getstate__(self):
#        tmp = {}
#        tmp['name'] = self.name
#        tmp['tree'] = self.tree
#        tmp['markers'] = self.markers
#        tmp['scatters'] = self.scatters
#        tmp['notes'] = self.notes
#        return tmp
        return self.__dict__

    def __setstate__(self, dict):
        for i in dict.keys():
            self.__dict__[i] = dict[i]


    def name_to_index(self, channels):
        """Return the channel indexes for the named channels"""

        if isinstance(channels,str):
            return self.channels.index(channels)
        idx = []
        for i in channels:
            try:
                idx.append(self.channels.index(i))
            except ValueError:
                try:
                    for j in range(1,int(self.notes.text['par'])+1):
                        if i == self.notes.text['p%dn' % j]:
                            idx.append(self.channels.index(self.notes.text['p%ds' % j]))
                except ValueError:
                    raise ValueError('%s is not in list' % i)
        if idx:
            return idx
        else:
            raise ValueError('field named a not found: %s' % str(channels))

    def get_channel_by_name(self, channels):
        """Return the data associated with specific channel names"""
        
        return self.tree.view()[:, self.name_to_index(channels)]

    def get_markers(self):
        """return the data associated with all the markers"""

        return self.view()[:, self.markers]

    def get_spill(self):
        """return the spillover matrix from the original fcs used in compisating"""
        
        try:
            return self.notes.text['spill']
        except KeyError:
            return None

    def view(self):
        """return the current view of the data"""
        
        return self.tree.view()

    def visit(self, name):
        """Switch current view of the data"""
        
        self.tree.visit(name)

    @property
    def current_node(self):
        """return the current node"""
        
        return self.tree.current

    def copy(self):
        """return a copy of fcm data object"""
        
        tname = self.name
        tpnts = self.tree.root.data
        tnotes = self.notes.copy()
        tchannels = self.channels[:]
        
        tscchannels = self.scatters[:]
        tmp = FCMdata(tname, tpnts, tchannels, tscchannels, tnotes)
        from copy import deepcopy
        tmp.tree = deepcopy(self.tree)
        return tmp

    def logicle(self, channels=None, T=262144, m=4.5, r=None, scale_max=1e5, scale_min=0):
        """return logicle transformed channels"""
        
        if channels is None:
            channels = self.markers
        return _logicle(self, channels, T, m, r, scale_max, scale_min)

    def hyperlog(self, channels, b, d, r, order=2, intervals=1000.0):
        """return hyperlog transformed channels"""
        
        return _hyperlog(self, channels, b, d, r, order, intervals)

    def log(self, channels=None):
        """return log base 10 transformed channels"""

        if channels is None:
            channels = self.markers        
        return _log(self, channels)

    def gate(self, g, chan=None):
        """return gated region of fcm data"""
        
        return g.gate(self, chan)

    def subsample(self, s):
        """return subsampled/sliced fcm data"""
        
        return s.subsample(self)
    
    def compensate(self, sidx=None, spill=None):
        '''Compensate the fcm data'''
        
        compensate(self,S=spill, markers=sidx)
        return self

    def get_cur_node(self):
        """ get current node """
        
        return self.current_node

    def add_view(self, node):
        """add a new node to the view tree"""
        
        self.tree.add_child(node.name, node)
        return self

    def summary(self):
        """returns summary of current view"""
        
        pnts = self.view()
        means = pnts.mean(0)
        stds = pnts.std(0)
        mins = pnts.min(0)
        maxs = pnts.max(0)
        medians = median(pnts, 0)
        dim = pnts.shape[1]
        summary = ''
        for i in range(dim):
            summary = summary + self.channels[i] + ":\n"
            summary = summary + "    max: " + str(maxs[i]) + "\n"
            summary = summary + "   mean: " + str(means[i]) + "\n"
            summary = summary + " median: " + str(medians[i]) + "\n"
            summary = summary + "    min: " + str(mins[i]) + "\n"
            summary = summary + "    std: " + str(stds[i]) + "\n"
        return summary

    def boundary_events(self):
        """returns dictionary of fraction of events in first and last
        channel for each channel"""
        
        boundary_dict = {}
        for k, chan in enumerate(self.channels):
            col = self.view()[:, k]
            boundary_dict[chan] = \
                sum((col == min(col)) | (col == max(col))) / len(col)
        return boundary_dict

    def export(self, file_name):
        '''
        export out current view to a fcs file
        '''
        from fcm.io import export_fcs
        export_fcs(file_name, self.view(), self.channels, self.notes.text)
