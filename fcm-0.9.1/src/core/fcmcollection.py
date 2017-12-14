"""
Data structure for a collection of FCMData objects.
All operations will be performed on each FCMData in the collection.
"""

from UserDict import DictMixin
from annotation import Annotation
import numpy

class FCMcollection(DictMixin):
    """
    Represent collection of FCMdata objects.
    Attributes: 
    note = collection level annotations
    tree = tree of operations
    """

    def __init__(self, name, fcms=None, notes=None):
        """
        Initialize with fcm collection and notes.
        """
        #  - how is this done in fcmdata?
        self.fcmdict = {}
        self.name = name
        if fcms is not None:
            for fcm in fcms:
                self.fcmdict[fcm.name] = fcm
        if notes is not None:
            self.notes = Annotation()
        else:
            self.notes = notes

    def keys(self):
        """
        D.keys() -> list of D's keys
        """
        return self.fcmdict.keys()

    def __getitem__(self, item):
        """return fcmcollection.fcmdict[item]"""
        return self.fcmdict[item]

    def __setitem__(self, key, value):
        """set fcmcollection.fcmdict[key] = value."""
        self.fcmdict[key] = value

    def __delitem__(self, key):
        """delete fcmcollection.fcmdict[key]"""
        del self.fcmdict[key]

    def __getattr__(self, name):
        """Convenience function to access fcm object by name."""
        if name in self.fcmdict.keys():
            return self.fcmdict[name]
        else:
            AttributeError("'%s' has no attribue '%s'" % (str(self.__class__), name))

    def check_names(self):
        """Checks for channel name consistency. 

        Returns dictionary of (fcmcollecion.name, [bool] | dictionary) where 
        True = all fcm have same name
        for the channel and False = at least one different name.
        """
        result_dict = {}
        results = []
        channels_list = []
        for item in self.values():
            if isinstance(item, self.__class__):
                results.append(item.check_names())
            else:
                channels_list.append(item.channels)
        name_tuples = zip(*channels_list)
        for name_tuple in name_tuples[:]:
            bits = []
            for name in name_tuple[1:]:
                if name == name_tuple[0]:
                    bits.append(True)
                else:
                    bits.append(False)
            results.append(
                reduce(numpy.logical_and, bits))
        result_dict[self.name] = results
        return result_dict
    
    def log(self, *args, **kwargs):
        """
        apply log transform the fcs objects in the collection
        """
        
        #TODO make it atomic?
        for i in self.fcmdict:
            self.fcmdict[i].log(*args,**kwargs)
        return self
    
    def logicle(self, *args, **kwargs):
        '''
        apply logicle transform to the fcs objects in the collection
        '''
        
        #TODO make it atomic?
        for i in self.fcmdict:
            self.fcmdict[i].logicle(*args, **kwargs)
        return self
    
    def compensate(self, *args, **kwargs):
        '''
        apply compensation to the fcs objects in a collection
        '''
        for i in self.fcmdict:
            self.fcmdict[i].compensate(*args, **kwargs)
        return self
    
    def gate(self, *args, **kwargs):
        '''
        apply a gate to the fcs objects in a collection
        '''
        for i in self.fcmdict:
            self.fcmdict[i].gate(*args, **kwargs)
        return self
    
    def summary(self):
        '''
        produce summary statitsics for each fcs object in the collection
        '''
        
        return '\n'.join(['%s:\n%s' % (i,self.fcmdict[i].summary()) for i in self.fcmdict])

    def classify(self, mixture):
        '''
        classify each fcs object in the collection using a mixture model
        '''
        
        rslt = {}
        for i in self.fcmdict:
            rslt[i] = mixture.classify(self.fcmdict[i])
        return rslt
    
    def fit(self, model, *args, **kwargs):
        '''
        fit a mixture model to each fcs object in a collection
        '''
        
        rslt = {}
        for i in self.fcmdict:
            rslt[i] = model.fit(self.fcmdict[i], *args, **kwargs)
        return rslt
    
    def to_list(self):
        '''
        return a list of the fcmdata objects contained in the collection
        '''
        
        return [self.fcmdict[i] for i in self.fcmdict]
    
    
if __name__ == '__main__':
    from io import loadFCS
    f1 = loadFCS('../../sample_data/3FITC_4PE_004.fcs')
    fs = FCMcollection([f1])

    print fs.keys()
    print fs.values()

