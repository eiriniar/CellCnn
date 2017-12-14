import numpy
#from matplotlib.nxutils import points_inside_poly
from tree import GatingNode


class Filter(object):
    """An object representing a gatable region"""

    def __init__(self, vert, channels, name=None):
        """
        vert = vertices of gating region
        channels = indices of channels to gate on.
        """
        self.vert = vert
        self.chan = channels
        if name is None:
            self.name = ""
        else:
            self.name = name 

    def gate(self, fcm, chan=None):
        """do the actual gating here."""
        pass
    
    def __repr__(self):
        return "%s(%s,%s,%s)" % (self.__class__, str(self.vert), str(self.chan), self.name)
    
    def __str__(self):
        return "<%s (%s) on %s>" % (self.__class__, self.name, str(self.chan))
        



class PolyGate(Filter):
    """
    An object representing a polygonal gatable region
    """
    def gate(self, fcm, chan=None, invert=False, name=None):
        """
        return gated region of FCM data
        """
        if chan is None:
            chan = self.chan
        if isinstance(chan, tuple):
            chan = list(chan)
        for i,j in enumerate(chan):
            if isinstance(j,str):
                chan[i] = fcm.name_to_index(j)[0]
        
        if name is None:
            name = self.name
        idxs = points_in_poly(self.vert, fcm.view()[:, chan])

        # matplotlib has points in poly routine in C
        # no faster than our numpy version
        # idxs = points_inside_poly(fcm.view()[:, chan], self.vert)

        if invert:
            idxs = numpy.invert(idxs)

        node = GatingNode(name, fcm.get_cur_node(), idxs)
        fcm.add_view(node)
        return fcm


class QuadGate(Filter):
    """
    An object to divide a region to four quadrants
    """
    def gate(self, fcm, chan=None, name=None, _full=False):
        """
        return gated region
        """
        if chan is None:
            chan = self.chan
        
        if name is None:
            if self.name is not '':
                name = self.name
            else:
                name = ''

            
        if name is not "" and len(name) != 4:
            raise ValueError('name must be empty or contain 4 items: name is %s' % str(name))
            
        
        # I (+,+), II (-,+), III (-,-), and IV (+,-)
        x = fcm.view()[:, chan[0]]
        y = fcm.view()[:, chan[1]]
        quad = {}
        quad[1] = (x > self.vert[0]) & (y > self.vert[1]) # (+,+)
        quad[2] = (x < self.vert[0]) & (y > self.vert[1]) # (-,+)
        quad[3] = (x < self.vert[0]) & (y < self.vert[1]) # (-,-)
        quad[4] = (x > self.vert[0]) & (y < self.vert[1]) # (+,-)
        root = fcm.get_cur_node()
        cname = root.name
        
        if name is "" :
            name = ["q%d"% i  for i in quad.keys()]
            
        if _full:
            nodes = []
        for i in quad.keys():
            if True in quad[i]:
                fcm.tree.visit(cname)
                node = GatingNode(name[i-1], root, quad[i])
                fcm.add_view(node)
                if _full:
                    nodes.append(node)
        if _full:
            return nodes
        else:
            return fcm

class IntervalGate(Filter):
    """
    An objeect to return events within an interval in any one channel.
    """
    def gate(self, fcm, chan=None, name=None):
        """
        return interval region.
        """
        if chan is None:
            chan = self.chan
            
        if name is None:
            name = self.name

        assert(len(self.chan) == 1)
        assert(len(self.vert) == 2)
        assert(self.vert[1] >= self.vert[0])

        x = fcm.view()[:, chan[0]]
        idxs = numpy.logical_and(x > self.vert[0], x < self.vert[1])

        node = GatingNode(name, fcm.get_cur_node(), idxs)
        fcm.add_view(node)
        return fcm

class ThresholdGate(Filter):
    """
    an object to return events above or below a threshold in any one channel
    """
    def __init__(self, vert, channels, op = 'g', name=None):
        """
        vert = boundry region
        channels = indices of channel to gate on.
        op = 'g' (greater) or 'l' (less) 
        """
        self.vert = vert
        self.chan = channels
        self.op = op
        
        if name is None:
            self.name = ""
        else:
            self.name = name
        
        
    def gate(self, fcm, chan=None, op=None, name=None):
        """
        return all events greater (or less) than a threshold
        allowed op are 'g' (greater) or 'l' (less)
        """
        if chan is None:
            chan = self.chan
            
        x = fcm.view()[:, chan]
        if op is None:
            op = self.op
            
        if op == 'g':
            idxs = numpy.greater(x,self.vert)
        elif op == 'l':
            idxs = numpy.less(x,self.vert)
        else:
            raise ValueError('op should be "g" or "l", received "%s"' % str(op))
        
        if name is None:
            name = self.name
            
        node = GatingNode(name, fcm.get_cur_node(), idxs)
        fcm.add_view(node)
        return fcm

def generate_f_score_gate(neg_smaple, pos_sample, chan, beta = 1, theta = 2, high=True):
    '''
    given a negative and a positive sample, calculate the 'optimal' threshold gate
    position from aproximate f-score calculation
    '''
    
    neg_hist, bins = numpy.histogram(neg_smaple[:,chan], 1000, normed=True)
    pos_hist, bins = numpy.histogram(pos_sample[:,chan], bins, normed=True)
    
    xs = (bins[1:] + bins[:-1])/2.0
    
    x0 = numpy.argmax(neg_hist)
    
    dfa = diff_pseudo_f1(neg_hist[x0:], pos_hist[x0:], beta=beta, theta = theta)
    
    f_cutoff = xs[x0+numpy.argmax(dfa)]
    
    if high:
        return ThresholdGate(f_cutoff, chan, 'g')
    else:
        return ThresholdGate(f_cutoff, chan, 'l')
    
    
    
def diff_pseudo_f(neg_pdf, pos_pdf, beta=1, theta=2, full=False):
    n = len(neg_pdf)
    c1 = numpy.array([numpy.sum(pos_pdf[i:]) for i in range(n)])
    c2 = numpy.array([numpy.sum(neg_pdf[i:]) for i in range(n)])
    c3 = numpy.where(pos_pdf > theta*neg_pdf, pos_pdf-neg_pdf, 0)
    while numpy.all(c3==0):
        theta -= 0.01
        c3 = numpy.where(pos_pdf > theta*neg_pdf, pos_pdf-neg_pdf, 0)
    c4 = numpy.array([numpy.sum(c3[i:]) for i in range(n)])
    precision = c1/(c1+c2)
    # recall = c1/numpy.sum(pos_pdf)
    recall = c4/numpy.sum(c3)
    diff = (1+beta*beta)*(precision*recall)/(beta*beta*precision + recall)
    if full:
        return precision, recall, diff
    else:
        return diff
    
def diff_pseudo_f1(neg_pdf, pos_pdf, beta=1, theta=2, full=False):
    n = len(neg_pdf)
    fpos = numpy.where(pos_pdf > theta*neg_pdf, pos_pdf-neg_pdf, 0)
    tp = numpy.array([numpy.sum(fpos[i:]) for i in range(n)])
    fn = numpy.array([numpy.sum(fpos[:i]) for i in range(n)])
    fp = numpy.array([numpy.sum(neg_pdf[i:]) for i in range(n)])
    precision = tp/(tp+fp)
    precision[tp==0]=0
    recall = tp/(tp+fn)
    recall[recall==0]=0
    diff = (1+beta*beta)*(precision*recall)/(beta*beta*precision + recall)

    if full:
        return precision, recall, diff
    else:
        return diff

    
def scale(k=1):
    """Closure to generate rescaling function with min=0 and max=k."""
    def f(x):
        _x = numpy.array(x)
        return k * (_x - numpy.min(_x))/(numpy.max(_x) - numpy.min(_x))
    return f

def points_in_poly(vs, ps):
    """Return boolean index of events from ps that are inside polygon with vertices vs.

    vs = numpy.array((k, 2))
    ps = numpy.array((n, 2))
    """

    # optimization to check only events within bounding box
    # for polygonal gate - useful if gating region is small
    # area_ratio_threshold = 0.5
    # area_gate_bb = numpy.prod(numpy.max(vs, 0) - numpy.min(vs, 0))
    # area_gate_ps = numpy.prod(numpy.max(ps, 0) - numpy.min(ps, 0))
    # if area_gate_bb/area_gate_ps < area_ratio_threshold:
    #     idx = numpy.prod((ps > numpy.min(vs, 0)) & (ps < numpy.max(vs, 0)),1)
    #     ps = ps[idx.astype('bool'), :]

    j = len(vs) - 1
    inPoly = numpy.zeros((len(vs), len(ps)), 'bool')

    for i, v in enumerate(vs):
        inPoly[i, :] |= ((v[0] < ps[:, 0]) & (vs[j, 0] >= ps[:, 0])) | ((vs[j, 0] < ps[:, 0]) & (v[0] >= ps[:, 0]))
        inPoly[i, :] &= (v[1] + (ps[:, 0] - v[0]) / (vs[j, 0] - v[0]) * (vs[j, 1] - v[1]) < ps[:, 1])
        j = i

    return numpy.bitwise_or.reduce(inPoly, 0)

if __name__ == '__main__':
    vertices = numpy.array([[2, 2], [10, 2], [10, 10], [2, 10]], 'd')
    points = numpy.random.uniform(0, 10, (10000000, 2))

    import time
    start = time.clock()
    inside = points_in_poly(vertices, points)
    print "Time elapsed: ", time.clock() - start
    print numpy.sum(inside)

