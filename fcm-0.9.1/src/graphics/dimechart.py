import numpy
import pylab

def dimechart(indicators, values, labels,
              cmap=pylab.cm.gist_rainbow, xlab='', ylab=''):
    indicators = numpy.array(indicators)
    values = numpy.array(values)
    k, p = values.shape
    w, h = 1, 10
    cs = cmap(numpy.linspace(0, 1, p))

    pylab.figure(figsize=(p, 3 * k))

    axs = []
    # keys
    pylab.subplot(k + 1, 1, 1)
    for _p in range(p):
        xpts = (2 + p) * numpy.array([_p, _p + 1, _p + 1, _p])
        ypts = 10 * numpy.array([0, 0, h, h])
        pylab.fill(xpts, ypts,
                   color=cs[_p], alpha=0.5, closed=True,
                   ec='k')
        pylab.text((2 + p) * (_p + w / 2.), 10 * h / 2, labels[_p],
                   va='center', ha='center',
                   fontsize=16, rotation=90)
    pylab.xticks([])
    pylab.yticks([])

    # data
    xtk = range(2 + p / 2, (2 + p) * (p + 1), p + 2)
    for _k in range(k):
        pylab.subplot(k + 1, 1, _k + 2)
        pylab.ylabel(ylab)
        if _k < (k - 1):
            pylab.xticks([])
        else:
            pylab.xticks(xtk, map(str, range(p, 0, -1)))
            pylab.xlabel(xlab)
        for _p in range(p):
            make_spark(indicators[_k, _p],
                       (2 + p) * _p,
                       100 * values[_k, _p],
                       w, h, cmap)
        axs.append(pylab.axis())

    axs = numpy.array(axs)
    xmin = min(axs[:, 0])
    xmax = max(axs[:, 1])
    xr = xmax - xmin
    unused_ymin = min(axs[:, 2])
    unused_ymax = max(axs[:, 3])

    for _k in range(k + 1):
        pylab.subplot(k + 1, 1, _k + 1)
        pylab.axis([xmin - xr / 20., xmax + xr / 20., -h - 2, 100 + h + 2])

    pylab.show()

def make_spark(xs, xoffset=0, yoffset=0, w=1, h=10,
               cmap=pylab.cm.gist_rainbow):
    # make central line
    n = len(xs)
    dw = (w - 1) / 2.0
    pylab.plot([xoffset + 1 - dw, xoffset + 1 + n + dw], [yoffset, yoffset], 'k-')
    cs = cmap(numpy.linspace(0, 1, n))
    for k, x in enumerate(xs):
        if x: # up
            xpts = numpy.array([k + 1 - dw, k + 2 + dw, k + 2 + dw, k + 1 - dw]) + xoffset
            ypts = numpy.array([0, 0, h, h]) + yoffset
            pylab.fill(xpts, ypts, color=cs[k], alpha=0.5, closed=True,
                       ec='k')
        else:
            xpts = numpy.array([k + 1 - dw, k + 2 + dw, k + 2 + dw, k + 1 - dw]) + xoffset
            ypts = numpy.array([0, 0, -h, -h]) + yoffset
            pylab.fill(xpts, ypts, color=cs[k], alpha=0.5, closed=True,
                       ec='k')

if __name__ == '__main__':
    i = numpy.random.randint(0, 2, (2, 10, 10))
    d = numpy.random.uniform(0, 1, (2, 10))
    l = map(str, range(10))
    print d
    dimechart(i, d, l, pylab.cm.gist_rainbow)
    pylab.show()

