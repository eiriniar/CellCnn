'''
Created on Aug 27, 2009

@author: jolly
'''
from tree import SubsampleNode, DropChannelNode



class Subsample(object):
    '''
    Takes a slice object and performs subsampling on the fcm object.
    '''


    def __init__(self, slicing):
        '''
        slicing = tuple of slices
        '''
        self.samp = slicing

    def subsample(self, fcm):
        """D(<fcmdata>) -> leads to a new fcm view of subsampled data"""
        node = SubsampleNode("", fcm.get_cur_node(), self.samp)
        fcm.add_view(node)
        return fcm

class _SubsampleFactory(object):
    '''
    factory generator of subsample objects
    '''

    def __init__(self):
        pass

    def __getitem__(self, item):
        return Subsample(item)

SubsampleFactory = _SubsampleFactory()


class DropChannel(object):
    """
    Drop channels by name from a fcm view
    """
    def __init__(self, idxs):
        self.idxs = idxs

    def drop(self, fcm):
        """D(<fcmdata>) -> create a new view in the fcm object missing the specified channels"""
        channels = fcm.channels[:]
        for i in self.idxs:
            channels.remove(i)
        left = []
        for i in channels:
            left.extend(fcm.name_to_index(i))

        left.sort()
        node = DropChannelNode("", fcm.get_cur_node(), left, channels)
        fcm.add_view(node)
        return fcm
