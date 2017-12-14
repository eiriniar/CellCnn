"""Collection of common FCM graphical plots."""

from util import bilinear_interpolate
from scipy import histogram
from numpy import histogram2d
import pylab

def hist(fcms, index, savefile=None, display=True, **kwargs):
    """Plot overlay histogram.

    fcms is a list of FCMData objects/arrays
    index is channel to plot
    """
    figure = pylab.figure()
    for fcm in fcms:
        if isinstance(index, str):
            index = fcm.name_to_index(index)
        y = fcm[:, index]
        h, b = histogram(y, bins=200, **kwargs)
        b = (b[:-1] + b[1:]) / 2.0
        unused_x = pylab.linspace(min(y), max(y), 100)
        pylab.plot(b, h, label=fcm.name)
        pylab.legend()

    if display:
        pylab.show()

    if savefile:
        pylab.savefig(savefile)

    return figure


def pseudocolor(fcm, indices, nrows=1, ncols=1, s=1, edgecolors='none',
            savefile=None, display=True,
            **kwargs):
    """Plot a pseudocolor.

    indices = list of marker index pairs
    nrows = number of rows to plot
    ncols = number of cols to plot
    nrows * ncols should be >= len(indices)
    display = boolean indicating whether to show plot
    save = filename to save plot (e.g. 'x.png')
    **kwargs are passed on to pylab.scatter
    """

    if nrows == 1 and ncols == 1:
        ncols = len(indices)

    assert(nrows * ncols >= len(indices))

    figure = pylab.figure(figsize=(ncols * 4, nrows * 4))

    for i, idx in enumerate(indices):
        pylab.subplot(nrows, ncols, i + 1)
        if (idx[0] != idx[1]):
            x = fcm[:, idx[0]]
            y = fcm[:, idx[1]]
            if 'c' not in kwargs:
                z = bilinear_interpolate(x, y)
                pylab.scatter(x, y, c=z, s=s, edgecolors=edgecolors, **kwargs)
            else:
                pylab.scatter(x, y, s=s, edgecolors=edgecolors, **kwargs)
            if isinstance(idx[0], str):
                pylab.xlabel(idx[0])
            else:
                pylab.xlabel(fcm.channels[idx[0]])
                
            if isinstance(idx[1], str):
                pylab.ylabel(idx[1])
            else:
                pylab.ylabel(fcm.channels[idx[1]])
        pylab.xticks([])
        pylab.yticks([])
        pylab.axis('equal')

    if display:
        pylab.show()

    if savefile:
        pylab.savefig(savefile)

    return figure

def pseudocolors(fcm, savefile=None, display=True, **kwargs):
    """PLot scatter matrix of all pseudocolors."""
    n = fcm.shape[1]
    indices = [(i, j) for i in range(n) for j in range(n)]
    pseudocolor(fcm, indices, nrows=n, ncols=n, savefile=savefile,
            display=display, **kwargs)

def pair_plot(data, savefile=None, display=True, **kwargs):
    chan = data.channels
    l = len(chan)
    figure = pylab.figure()
    pylab.subplot(l, l, 1)
    for i in range(l):
        for j in range(i + 1):
            pylab.subplot(l, l, i * l + j + 1)
            if i == j:
                pylab.hist(data[:, i], bins=200, histtype='stepfilled')
            else:
                pylab.scatter(data[:, i], data[:, j], **kwargs)

            if j == 0:
                pylab.ylabel(chan[i])
            if i == l - 1:
                pylab.xlabel(chan[j])

    if display:
        pylab.show()

    if savefile:
        pylab.savefig(savefile)

    return figure


def contour(data, indices, savefile=None, display=True, **kwargs):
    x = data[:, indices[0]]
    y = data[:, indices[1]]
    z = histogram2d(x, y)
    figure = pylab.figure()
    pylab.contour(z[0])


    if display:
        pylab.show()

    if savefile:
        pylab.savefig(savefile)

    return figure

if __name__ == '__main__':
    import sys
    sys.path.append('../')
    from io import FCSreader

    fcm = FCSreader('../../sample_data/3FITC_4PE_004.fcs').get_FCMdata()
    # pseudocolor(fcm, [(0,1),(2,3)], nrows=1, ncols=2, s=1, edgecolors='none')
    pseudocolors(fcm, s=1, edgecolors='none', display=False,
             savefile='3FITC_4PE_004.png', cmap=pylab.cm.hsv)



