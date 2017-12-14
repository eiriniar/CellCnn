'''
Graphical quadrant gating
'''
import matplotlib.pyplot as plt
import time
from fcm import QuadGate

class Gate(object):
    """Gate class implements gating using Matplotlib animation and events.

    Right click to add vertex.
    Left click and drag vertex to move.
    When vertices >= 3, polygonal gate will display.
    Double click within gate to extract gated events and zoom to gated region.
    """

    def __init__(self, fcm, idxs, ax):
        self.fcm = fcm
        self.idxs = idxs
        ax.scatter(fcm[:, idxs[0]], fcm[:, idxs[1]],
                   s=1, c='b', edgecolors='none')

        self.canvas = ax.figure.canvas
        self.ax = ax
        self.vertices = []
        self.poly = None
        self.background = None
        self.g = None

        self.t = time.time()
        self.double_click_t = 1.0
        self.cid_press = self.canvas.mpl_connect(
            'button_press_event', self.onclick)

        self.hline = None
        self.vline = None

    def onclick(self, event):
        xmin, xmax, ymin, ymax = self.ax.axis()
        #h = ymax - ymin
        w = xmax - xmin
        if event.button == 3:
            if self.hline:
                print 'moving lines'
                self.hline.set_xdata([event.xdata, event.xdata])
                self.vline.set_ydata([event.ydata, event.ydata])
            else:
                print 'setting up lines'
                self.hline = self.ax.plot([event.xdata, event.xdata], [ymin, ymax], linewidth=.001 * w, c='black')[-1]
                self.vline = self.ax.plot([xmin, xmax], [event.ydata, event.ydata], linewidth=.001 * w, c='black')[-1]

        if event.button == 1:
            if (time.time() - self.t < self.double_click_t):
                self.gate(event.xdata, event.ydata)


        self.t = time.time()
        self.update()

    def disconnect(self):
        'disconnect all the stored connection ids'
        self.canvas.mpl_disconnect(self.cid_press)

    def update(self):
        self.canvas.draw()

    def gate(self, x, y):
        g = QuadGate([x, y], self.idxs)
        self.fcm.gate(g)
        self.g = g

def quad_gate(fcm, idxs):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    gate = Gate(fcm, idxs, ax)
    plt.show()
    return gate.g


if __name__ == '__main__':
    import sys
    sys.path.append('../')
    from io import FCSreader
    #import networkx

    fcm = FCSreader('../../sample_data/3FITC_4PE_004.fcs').get_FCMdata()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    idxs = [2, 3]
    gate = Gate(fcm, idxs, ax)

    plt.show()
    plt.show()
    print fcm.tree.view()
    print fcm.tree.nodes
