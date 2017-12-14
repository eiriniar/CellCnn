'''
Created on Feb 11, 2012

@author: Jacob Frelinger
'''

import xml.etree.cElementTree as xml
import numpy
from fcm import PolyGate
from fcm.io.readfcs import is_fl_channel
from fcm.core.transforms import _logicle
from collections import namedtuple

FlatGate = namedtuple('FlatGate', ['gate','parent'])
        

class PopulationNode(object):
    '''
    node for gates in xml tree
    '''
    def __init__(self, name, gate, subpops = None):
        self.name = name
        self.gate = gate
        if subpops is None:
            self.subpops = {}
        else:
            self.subpops = subpops
        
    @property
    def gates(self):
        a = [self.gate]
        for i in self.subpops:
            a.extend([self.subpops[i].gates])
        if len(a) == 1:
            return a[0]
        else:
            return a
    

    def pprint(self, depth=0):
        j = "  " * depth + self.name + "\n"
        for i in self.subpops:
            j += self.subpops[i].pprint(depth+1)            
        return j    
    
    def apply_gates(self, file):
        #print self.gate.name, self.gate.chan
        self.gate.gate(file)
        node = file.current_node
        for i in self.subpops:
            file.visit(node)
            self.subpops[i].apply_gates(file)
        
    def logicle(self, T=262144, m=4.5, r=None, w=0.5, scale_max=1e5):
        '''
        convert gate cordinates to logicle scale from linear scale
        '''
        x_chan, y_chan = self.gate.chan
        xs = numpy.array([i[0] for i in self.gate.vert])
        ys = numpy.array([i[1] for i in self.gate.vert])
        if is_fl_channel(x_chan):
            xs = scale_max*_logicle(xs, T, m, r, w)
        if is_fl_channel(y_chan):
            ys = scale_max*_logicle(ys, T, m, r, w)
        self.gate.vert = zip(xs,ys)
        print zip(xs,ys)
        for i in self.subpops:
            self.subpops[i].logicle(T,m,r,w,scale_max)
    
class xmlfcsfile(object):
    '''
    container object for fcs file defined in a flowjo xml file
    '''
    def __init__(self, name, comp=None, pops = None):
        self.name = name
        if pops is None:
            self.pops = {}
        else:
            self.pops = pops
        self.comp = comp
        
    
    @property
    def gates(self):
        a = []
        for i in self.pops:
            a.append(self.pops[i].gates)
        
        return a
    
    
    def flat_gates(self, a=None, parent=None):
        if a is None:
            a = self.gates
            parent = 'root'
        last_parent=False
        flat = []
        for i in a:
            if isinstance(i,list):
                flat.extend(self.flat_gates(i, parent))
            else:
                flat.append(FlatGate(i,parent))
                if not last_parent:
                    parent = i.name
                    last_parent = True
            
        return flat
        
    def pprint(self, depth=0):
        j = "  " * depth + self.name + "\n"
        if self.pops is not None:
            for i in self.pops:
                j += self.pops[i].pprint(depth+1)
            
        return j
    
    def apply_gates(self, file):
        node = file.current_node
        for i in self.pops:
            file.visit(node)
            self.pops[i].apply_gates(file)
            
    def logicle(self, T=262144, m=4.5, r=None, w=0.5, scale_max=1e5):
        '''
        convert gate cordinates to logicle scale from linear scale
        '''
        for i in self.pops:
            self.pops[i].logicle(T,m,r,w,scale_max)
            
            
class FlowjoWorkspace(object):
    '''
    Object representing the files, gates, and compensation matricies from a 
    flowjo xml worksapce
    '''


    def __init__(self,tubes, comp=None):
        '''
        gates - dictionary of gates keyed by filename,
        comp - dictionary of defined compensation channels and matricies, keyed by name defaults to none
        '''
        
        self.tubes = tubes
        self.comp = comp
        
    @property
    def file_names(self):
        return [self.tubes[tube].name for tube in self.tubes]
    
    @property
    def gates(self):
        gates = {}
        for tube_name in self.tubes:
            tube = self.tubes[tube_name]
            gates[tube_name] = tube.gates
        return gates

    def flat_gates(self):
        gates = {}
        for tube_name in self.tubes:
            tube = self.tubes[tube_name]
            gates[tube_name] = tube.flat_gates()
        return gates
    
    def pprint(self):
        j = ''
        for i in self.tubes:
            j += self.tubes[i].pprint(0)
        
        return j
    
    def logicle(self, T=262144, m=4.5, r=None, w=0.5, scale_max=1e5):
        '''
        convert gate cordinates for all tubes into logicle scale from linear scale
        '''
        for i in self.tubes:
            self.tubes[i].logicle(T,m,r,w,scale_max)
            
def load_flowjo_xml(fh):
    '''
    create a FlowjoWorkspace object from a xml file
    '''
    if isinstance(fh,str):
        fh = open(fh,'r')
    
    tree = xml.parse(fh)
    
    root = tree.getroot()
    files = {}
    chans = []
    comps = {}
    prefix = ''
    suffix = ''
    psdict = {}
    for mats in root.iter('CompensationMatrices'):
        for mat in mats.iter('CompensationMatrix'):
            prefix = mat.attrib['prefix']
            suffix = mat.attrib['suffix']
            
            a = len([ i for i in mat.iter('Channel')])
            comp = numpy.zeros((a,a))
            chans = []
            for i,chan in enumerate(mat.iter('Channel')):
                chans.append(chan.attrib['name'])
                for j,sub in enumerate(chan.iter('ChannelValue')):
                    comp[i,j] = float(sub.attrib['value'])
            comps[mat.attrib['name']] = (chans,comp)
            psdict[mat.attrib['name']] = (prefix, suffix)
           
    for node in root.iter('Sample'):
        # pull out comp matrix
        keywords = node.find('Keywords')
        comp_matrix = ''
        if keywords is not None:
            
            for keyword in keywords.iter('Keyword'):
                if 'name' in keyword.attrib:
                    if keyword.attrib['name'] == 'FJ_CompMatrixName':
                        if keyword.attrib['value'] in comps:
                            comp_matrix = comps[keyword.attrib['value']]
                            
                        else:
                            comp_matrix = None
                        if keyword.attrib['value'] in psdict:
                            prefix, suffix = psdict[keyword.attrib['value']]
                        else:
                            prefix, suffix = (None, None)

        sample_name = None
        keywords = node.find('Keywords')
        if keywords is not None:
            for keyword in keywords.iter('Keyword'):
                if 'name' in keyword.attrib:
                    if keyword.attrib['name'] == '$FIL':
                        sample_name = keyword.attrib['value']

        # fall back on name attrib in SampleNode if above fails
        sample = node.find('SampleNode')
        if sample_name is None:
            sample_name = sample.attrib['name']
            
        if comp_matrix: 
            fcsfile = xmlfcsfile(sample_name, comp = comp_matrix)
        else:
            fcsfile = xmlfcsfile(sample_name)                      

        # if comp_matrix: 
        #     fcsfile = xmlfcsfile(sample.attrib['name'], comp = comp_matrix)
        # else:
        #     fcsfile = xmlfcsfile(sample.attrib['name'])                      

        # find gates
        fcsfile.pops = find_pops(sample, prefix, suffix)
        
        uniq = False
        count = 0
        # new_name = fcsfile.name
        
        while not uniq:
            if  sample_name in files:
                count += 1
                sample_name = fcsfile.name + '_%d' % count
            else:
                uniq = True
        files[sample_name] = fcsfile

                            
    if len(comps) > 0:
        return FlowjoWorkspace(files,comps)
    else:
        return FlowjoWorkspace(files)
 
 
def find_pops(node, prefix=None, suffix=None):
    pops = {}
    for i in node:
        if i.tag == 'Population':
            pops[i.attrib['name']] = build_pops(i, prefix, suffix)
    return pops
    
    
def build_pops(node,prefix=None, suffix=None, name_prefix=None):
    if isinstance(name_prefix, str):
        name = name_prefix + "::" + node.attrib['name']
    else:  
        name = node.attrib['name']
    
    children = {}
    for i in node:
        if i.tag == 'Population':
            tmp = build_pops(i, prefix, suffix, name_prefix=name)
            if tmp is not None:
                children[i.attrib['name']] = tmp
            
        elif i.tag == 'PolygonGate':
            for j in i:
                if j.tag == 'PolyRect':
                    g = build_Polygon(j, prefix, suffix, name_prefix)
                    
                elif j.tag == 'Polygon':
                    g = build_Polygon(j, prefix, suffix, name_prefix)
                    
    try:            
        return  PopulationNode(name, g, children)
    except UnboundLocalError:
        return None
    

def build_Polygon(rect, prefix=None, suffix=None, name_prefix=None):
    if isinstance(name_prefix, str):
        name = name_prefix + "::" + rect.attrib['name']
    else:  
        name = rect.attrib['name']
       
    verts = []
    axis = [rect.attrib['xAxisName'], rect.attrib['yAxisName']]

    if prefix is not None and prefix is not '':
        for i,j in enumerate(axis):
            if j.startswith(prefix):
                axis[i] = j.replace(prefix,'')
                
    if suffix is not None and suffix is not '':
        for i,j in enumerate(axis):
            if j.endswith(suffix):
                axis[i] = j[:-(len(suffix))]
                #replaced[i] = True
    axis = tuple(axis)

    for vert in rect.iter('Vertex'):
        x = float(vert.attrib['x'])
        y = float(vert.attrib['y'])
        verts.append((x,y))

    return PolyGate(verts, axis, name)

                    
    
if __name__ == "__main__":
    import fcm
    import sys
#    a = load_flowjo_xml('/home/jolly/Projects/fcm/scratch/flowjoxml/pretty.xml')
#    #print a.file_names
#    #print a.gates
#    foogate = a.gates['Specimen_001_A1_A01.fcs'][0][0]
#    cdfgate = a.gates['Specimen_001_A1_A01.fcs'][0][3]
#    cdegate = a.gates['Specimen_001_A1_A01.fcs'][0][6]
#    
#    #print a.comp.keys(), '\n', a.comp['Comp Matrix']
#    sidx, spill = a.comp['Comp Matrix']
#    #print a.tubes['Specimen_001_A1_A01.fcs'].comp[1] - a.comp['Comp Matrix'][1]
#    #print a.tubes['Specimen_001_A1_A01.fcs'].pprint()
#    x = fcm.loadFCS('/home/jolly/Projects/fcm/scratch/flowjoxml/001_05Aug11.A01.fcs', auto_comp=False, transform=None)#, sidx=sidx, spill=spill, transform='logicle')
#    #x.logicle()
#    x.compensate(sidx=sidx,spill=spill)
#    #print x.channels
#    a.tubes['Specimen_001_A1_A01.fcs'].apply_gates(x)
#    #print x.tree.pprint(size=True)
#    x.visit('foo1')
#    print x.current_node.name, x.shape, 238712
#    x.visit('foo1::cd4')
#    print x.current_node.name, x.shape, 43880
#    
#    x.visit('c1')
##    import pylab
##    pylab.subplot(2,2,1)
##    xx,yy = foogate.chan
##    pylab.xlabel(xx)
##    pylab.ylabel(yy)
##    pylab.scatter(x[xx],x[yy], s=1, edgecolor='none', c='grey')
#
##    z = numpy.array(foogate.vert)
##    pylab.fill(z[:,0], z[:,1], alpha=.2)
##    for vert in foogate.vert:
##        pylab.scatter(vert[0],vert[1], c='r')
##    x.visit('foo1')
##    pylab.scatter(x[xx],x[yy], s=1, edgecolor='none', c='blue')
#        
##    pylab.subplot(2,2,2)
##    xx,yy = cdfgate.chan
##    pylab.xlabel(xx)
##    pylab.ylabel(yy)
##    pylab.scatter(x[xx],x[yy], s=1, edgecolor='none', c='grey')
##    z = numpy.array(cdfgate.vert)
##    pylab.fill(z[:,0], z[:,1], alpha=.2)
##    for vert in cdfgate.vert:
##        pylab.scatter(vert[0],vert[1], c='r')
#    x.visit('foo1::cd4')
##    pylab.scatter(x[xx],x[yy], s=1, edgecolor='none', c='blue')
#    
#    from fcm.core.transforms import _logicle
#    x.visit('foo1')
##    nxx = 10**5*_logicle(x[xx][:,0],262144,4.5,None,0.5)
##    nyy = 10**5*_logicle(x[yy][:,0],262144,4.5,None,0.5)
#    #nxx[nxx<0] = 0
#    #nyy[nyy<0] = 0
#    verts = cdfgate.vert
#    verts = [10**5*_logicle(numpy.array(i),262144,4.5,None,0.5) for i in verts]
##    pylab.subplot(2,2,4)
##    z = numpy.array(verts)
##    pylab.fill(z[:,0], z[:,1], alpha=.2)
##    pylab.scatter(nxx,nyy,s=1, edgecolor='none',c='grey')
##    for i in verts:
##        pylab.scatter(i[0],i[1], c='r')
##    
##    ax = pylab.gca()
##    import fcm.graphics
##    fcm.graphics.set_logicle(ax,'x')
##    fcm.graphics.set_logicle(ax,'y')
#    cdfgate.vert = verts
#    print verts
#    cdfgate.name = 'cd4 transform'
#    x.logicle()
#    x.gate(cdfgate)
#    
#    verts = cdegate.vert
#    print verts
#    verts = [10**5*_logicle(numpy.array(i),262144,4.5,None,0.5) for i in verts]
#    print verts
#    #sys.exit()
#    cdegate.vert = verts
#    cdegate.name = 'cd8 transform'
#    
#    x.visit('t1')
#    x.gate(cdegate)
#    
##    pylab.show()
##    
#    print x.tree.pprint(size=True)
##
##    x = fcm.loadFCS('/home/jolly/Projects/fcm/scratch/flowjoxml/001_05Aug11.A02.fcs', auto_comp=False, transform=None)#, sidx=sidx, spill=spill, transform='logicle')
##    #x.logicle()
##    x.compensate(sidx=sidx,spill=spill)
##    #print x.channels
##    a.tubes['Specimen_001_A2_A02.fcs'].apply_gates(x)
##    #print x.tree.pprint(size=True)
##    print 'NEW'
##    x.visit('c1')
##    
##    norm = float(x.shape[0])
##    x.visit('foo1')
##    print x.current_node.name, x.shape, x.shape[0]/norm
##    norm = float(x.shape[0])
##    x.visit('cd4')
##    print x.current_node.name, x.shape, x.shape[0]/norm
##    x.visit('cd8')
##    print x.current_node.name, x.shape, x.shape[0]/norm

    print "NEW"
    a = load_flowjo_xml('/home/jolly/Projects/fcm/scratch/flowjoxml/pretty.xml')
    print type(a)
    print type(a.tubes['Specimen_001_A1_A01.fcs'])
    for i in  a.tubes['Specimen_001_A1_A01.fcs'].flat_gates():
        print i.parent, '<==',i.gate.name
        
    print a.flat_gates()
#    a.logicle()
#    sidx, spill = a.comp['Comp Matrix']
#    
#    x = fcm.loadFCS('/home/jolly/Projects/fcm/scratch/flowjoxml/001_05Aug11.A01.fcs', sidx=sidx, spill=spill, transform='logicle')
#
#    
#    a.tubes['Specimen_001_A1_A01.fcs'].apply_gates(x)
#    print x.tree.pprint(size=True)
