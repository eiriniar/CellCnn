from __future__ import division
import numpy
import math
import pylab
from fcm.core.transforms import _logicle as logicle
# from scipy.interpolate import interp2d

def bilinear_interpolate(x, y, bins=None):
    """Returns interpolated density values on points (x, y).
    
    Ref: http://en.wikipedia.org/wiki/Bilinear_interpolation.
    """
    if bins is None:
        bins = int(numpy.sqrt(len(x)))

    z, unused_xedge, unused_yedge = numpy.histogram2d(y, x, bins=[bins, bins],
                                        range=[(numpy.min(y), numpy.max(y)),
                                               (numpy.min(x), numpy.max(x))]
                                        )
    xfrac, xint = numpy.modf((x - numpy.min(x)) /
                             (numpy.max(x) - numpy.min(x)) * (bins - 1))
    yfrac, yint = numpy.modf((y - numpy.min(y)) /
                             (numpy.max(y) - numpy.min(y)) * (bins - 1))

    xint = xint.astype('i')
    yint = yint.astype('i')

    z1 = numpy.zeros(numpy.array(z.shape) + 1)
    z1[:-1, :-1] = z

    # values at corners of square for interpolation
    q11 = z1[yint, xint]
    q12 = z1[yint, xint + 1]
    q21 = z1[yint + 1, xint]
    q22 = z1[yint + 1, xint + 1]

    return q11 * (1 - xfrac) * (1 - yfrac) + q21 * (1 - xfrac) * (yfrac) + \
        q12 * (xfrac) * (1 - yfrac) + q22 * (xfrac) * (yfrac)

def trilinear_interpolate(x, y, z, bins=None):
    """Returns interpolated density values on points (x, y, z).
    
    Ref: http://en.wikipedia.org/wiki/Trilinear_interpolation.
    """
    if bins is None:
        bins = int(len(x) ** (1 / 3.0))

    vals = numpy.zeros((len(x), 3), 'd')
    vals[:, 0] = x
    vals[:, 1] = y
    vals[:, 2] = z

    h, unused_edges = numpy.histogramdd(vals,
                                 bins=[bins, bins, bins]
                                 )
    xfrac, xint = numpy.modf((x - numpy.min(x)) /
                             (numpy.max(x) - numpy.min(x)) * (bins - 1))
    yfrac, yint = numpy.modf((y - numpy.min(y)) /
                             (numpy.max(y) - numpy.min(y)) * (bins - 1))
    zfrac, zint = numpy.modf((z - numpy.min(z)) /
                             (numpy.max(z) - numpy.min(z)) * (bins - 1))

    xint = xint.astype('i')
    yint = yint.astype('i')
    zint = zint.astype('i')

    h1 = numpy.zeros(numpy.array(h.shape) + 1)
    h1[:-1, :-1, :-1] = h

    # values at corners of cube for interpolation
    q111 = h1[xint, yint, zint]
    q112 = h1[xint + 1, yint, zint]
    q122 = h1[xint + 1, yint + 1, zint]
    q121 = h1[xint, yint + 1, zint]
    q211 = h1[xint, yint, zint + 1]
    q212 = h1[xint + 1, yint, zint + 1]
    q222 = h1[xint + 1, yint + 1, zint + 1]
    q221 = h1[xint, yint + 1, zint + 1]

    i1 = q111 * (1 - zfrac) + q211 * (zfrac)
    i2 = q121 * (1 - zfrac) + q221 * (zfrac)
    j1 = q112 * (1 - zfrac) + q212 * (zfrac)
    j2 = q122 * (1 - zfrac) + q222 * (zfrac)

    w1 = i1 * (1 - yfrac) + i2 * (yfrac)
    w2 = j1 * (1 - yfrac) + j2 * (yfrac)

    return w1 * (1 - xfrac) + w2 * (xfrac)


def color_map(nclusts):
    '''
    return a list of rgb values spaced over the color wheel.
    '''
    return [ floatRgb(i, 0, nclusts + 1) for i in range(nclusts + 1)]

def floatRgb(mag, cmin, cmax, alpha=1.0):
    """
    Return a tuple of floats between 0 and 1 for the red, green and
    blue amplitudes.
    """

    try:
        # normalize to [0,1]
        x = float(mag - cmin) / float(cmax - cmin)
    except:
        # cmax = cmin
        x = 0.5
    blue = min((max((4 * (0.75 - x), 0.)), 1.))
    red = min((max((4 * (x - 0.25), 0.)), 1.))
    green = min((max((4 * math.fabs(x - 0.5) - 1., 0.)), 1.))
    return (red, green, blue, alpha)


def plot_mu_labels(mu, colors, dims):
    x, y = dims
    for j in range(mu.shape[0]):
        pylab.text(mu[j, x], mu[j, y], str(j), fontsize=12, weight='bold',
                                                  bbox=dict(facecolor=colors[j], alpha=0.5),
                                                  va='center', ha='center')

def set_logicle(ax, xy, T =262144, m=4.5, w=0.5, scale_max = 10**5):
    scale = scale_max*logicle(numpy.array([0, 100, 10**3, 10**4, 10**5]), T, m, None, w)
    labels = ['0', '10^2', '10^3', '10^4', '10^5']
    
    minorraw = numpy.hstack([numpy.linspace(10**i,10**(i+1), 10) for i in range(2,5)])
    minorvalues = scale_max*logicle(minorraw, T, m, None, w)
    if xy == 'x':
        ax.set_xticks(scale)
        ax.set_xticklabels(labels)
        ax.set_xticks(minorvalues, minor=True)
    elif xy == 'y':
        ax.set_yticks(scale)
        ax.set_yticklabels(labels)
        ax.set_yticks(minorvalues, minor=True)
        
    else:
        raise TypeError('Unknown axis to label "%s"' % str(xy))
    

if __name__ == '__main__':
    import sys
    sys.path.append('../')
    from io import FCSreader

    fcm = FCSreader('../../sample_data/3FITC_4PE_004.fcs').get_FCMdata()
    x = fcm[:, 0]
    y = fcm[:, 1]

    import time
    start = time.clock()
    z = bilinear_interpolate(x, y)
    print time.clock() - start

    pylab.scatter(x, y, s=1, c=z, edgecolors='none', cmap=pylab.cm.get_cmap())
    pylab.show()
