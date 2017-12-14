from util import bilinear_interpolate
from fcm import PolyGate, IntervalGate, ThresholdGate, QuadGate

def plot_gate(data, gate, ax, chan=None, name=None, **kwargs):
    """
    wrapper around several plots aimed at plotting gates (and gating at the same time)
    see the wrapped functions plot_ploy_gate, plot_threshold_gate, plot_threshold_hist
    for more information
    """
    if isinstance(gate,PolyGate):
        plot_poly_gate(data, gate, ax, chan, name, **kwargs)
    elif isinstance(gate,ThresholdGate):
        if isinstance(chan,int) or chan is None:
            plot_threshold_hist(data, gate, ax, chan, name, **kwargs)
        else:
            plot_threshold_gate(data, gate, ax, chan, name, **kwargs)
    elif isinstance(gate,QuadGate):
        plot_quad_gate(data, gate, ax, chan, name, **kwargs)
            
def plot_quad_gate(data, gate, ax, chan=None, name=None, **kwargs):
    if chan is None:
        chan = gate.chan
    node = data.get_cur_node()
    nodes = gate.gate(data, chan=chan, name=name, _full=True)
    colors = ['b','g','r','c']
    for i,node in enumerate(nodes):
        x = node.view()
        
        if x.shape>0:
            ax.scatter(x[:,chan[0]],x[:,chan[1]], s=1, edgecolor='none', c=colors[i])
    
    ax.axvline(gate.vert[0])
    ax.axhline(gate.vert[1])
        
def plot_threshold_gate(data, gate, ax, chan=None, name=None, **kwargs):
   
    if 'bgc' in kwargs:
        bgc = kwargs['bgc']
        del kwargs['bgc']
    else:
        bgc = 'grey'
        
    if 'bgalpha' in kwargs:
        bga = kwargs['bgalpha']
        del kwargs['bgalpha']
    else:
        bga = 1

    if 'c' in kwargs:
        z = kwargs['c']
        del kwargs['c'] # needed incase bgc is used
        calc_z = False
    else:
        calc_z = True
        
    if 'alpha' in kwargs:
        alpha = kwargs['alpha']
        del kwargs['alpha']
    else:
        alpha = 1 # is there a way to look up what this defaults to incase this is changed in .matplotlibrc?

    # add in support for specifing size
    
    ax.scatter(data[:,chan[0]],data[:,chan[1]], c=bgc, s=1, edgecolor='none', alpha=bga)
    gate.gate(data, name=name)
    
    #has to be set after gating...

    if data.shape[0] > 0:
        if calc_z:
            z = bilinear_interpolate(data[:, chan[0]], data[:, chan[1]])  

        ax.scatter(data[:,chan[0]],data[:,chan[1]], c=z, s=1, edgecolor='none', alpha=alpha, **kwargs)
    
    if chan[0] == gate.chan:
        ax.axvline(gate.vert)
    elif chan[1] == gate.chan:
        ax.axhline(gate.vert)


def plot_threshold_hist(data, gate, ax, chan=None, name=None, **kwargs):
    if 'bgcolor' in kwargs:
        bgc = kwargs['bgcolor']
        del kwargs['bgcolor']
    else:
        bgc = 'grey'
        
    if 'bgalpha' in kwargs:
        bga = kwargs['bgalpha']
        del kwargs['bgalpha']
    else:
        bga = 1

    if 'color' in kwargs:
        z = kwargs['color']
        del kwargs['color'] # needed incase bgc is used
    else:
        z = 'blue'
        
    if 'alpha' in kwargs:
        alpha = kwargs['alpha']
        del kwargs['alpha']
    else:
        alpha = 1 # is there a way to look up what this defaults to incase this is changed in .matplotlibrc?

    if 'bins' not in kwargs:
        kwargs['bins'] = 1000
        
    if 'normed' not in kwargs:
        kwargs['normed'] = True
        
#    if 'edgecolor' not in kwargs:
#        kwargs['edgecolor'] = 'none'
 
    if 'histtype' not in kwargs:
        kwargs['histtype'] = 'step'

    if chan is None:
        chan = gate.chan

    count, bins, patch = ax.hist(data[:,chan], edgecolor=bgc, alpha=bga, **kwargs)
    
    gate.gate(data, name=name)
    
    if 'bins' in kwargs:
        del kwargs['bins']

    count, bins, patch = ax.hist(data[:,chan], bins=bins, edgecolor=z, alpha=alpha, **kwargs)
    ax.axvline(gate.vert)


def plot_poly_gate(data, gate, ax, chan=None, name=None, **kwargs):
    if chan is None:
        chan = gate.chan
        
    if 'bgc' in kwargs:
        bgc = kwargs['bgc']
        del kwargs['bgc']
    else:
        bgc = 'grey'
        
    if 'bgalpha' in kwargs:
        bga = kwargs['bgalpha']
        del kwargs['bgalpha']
    else:
        bga = 1

    if 'c' in kwargs:
        z = kwargs['c']
        del kwargs['c'] # needed incase bgc is used
        calc_z = False
    else:
        calc_z = True
        
    if 'alpha' in kwargs:
        alpha = kwargs['alpha']
        del kwargs['alpha']
    else:
        alpha = 1 # is there a way to look up what this defaults to incase this is changed in .matplotlibrc?

    # add in support for specifing size
    
    ax.scatter(data[:,chan[0]],data[:,chan[1]], c=bgc, s=1, edgecolor='none', alpha=bga)
    gate.gate(data, chan=chan, name=name)

    #has to be set after gating...

    
    if data.shape[0] > 0:
        if calc_z:
            z = bilinear_interpolate(data[:, chan[0]], data[:, chan[1]])  

        ax.scatter(data[:,chan[0]],data[:,chan[1]], c=z, s=1, edgecolor='none', alpha=alpha, **kwargs)
    
    ax.fill(gate.vert.T[0], gate.vert.T[1], edgecolor='black', facecolor='none') 
    
    
if __name__ == '__main__':
    import fcm
    import numpy
    import matplotlib
    import matplotlib.pyplot as plt
    x = fcm.loadFCS('../../sample_data/3FITC_4PE_004.fcs')
    g = PolyGate(numpy.array([[0,0],[500,0],[500,500],[0,500]]), [0,1])
    
    g3 = QuadGate([250,300],(2,3))
    fig = plt.figure()
    ax = fig.add_subplot(2,2,1)
    plot_gate(x,g,ax, name="firstgate", alpha=.5, bgalpha=.5)
    ax = fig.add_subplot(2,2,2)
    mx = x[:,2].max()
    print mx
    g2 = ThresholdGate(mx-1,2)
    plot_gate(x,g2,ax, name="secondgate", chan=(2,3), alpha=.5)#, bgc='red', c='green')
    print x.shape
    print x[:]
    x.visit('root')
    ax = fig.add_subplot(2,2,3)
    plot_gate(x,g3,ax, name=['a','b','c','d'])
    print x.tree.pprint()
    plt.show()
    
