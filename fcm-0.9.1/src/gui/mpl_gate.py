import numpy
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
from fcm import PolyGate as g
from matplotlib.nxutils import points_inside_poly
import time

class DraggableVertex(object):
    """
    Draggable vertex of gating polygon.
    """

    lock = None  # only one can be animated at a time
    def __init__(self, circle, parent):
        self.parent = parent
        self.circle = circle
        self.press = None
        self.background = None
        self.canvas = self.circle.figure.canvas
        self.ax = self.circle.axes

    def connect(self):
        'connect to all the events we need'
        self.cid_press = self.canvas.mpl_connect(
            'button_press_event', self.on_press)
        self.cid_release = self.canvas.mpl_connect(
            'button_release_event', self.on_release)
        self.cid_motion = self.canvas.mpl_connect(
            'motion_notify_event', self.on_motion)

    def on_press(self, event):
        'on button press we will see if the mouse is over us and store some data'
        if event.inaxes != self.circle.axes: return
        if DraggableVertex.lock is not None: return
        contains, unused = self.circle.contains(event)
        if not contains: return
        # print 'event contains', self.circle.center
        x0, y0 = self.circle.center
        self.press = x0, y0, event.xdata, event.ydata
        DraggableVertex.lock = self
        self.parent.update()

    def on_motion(self, event):
        'on motion we will move the circle if the mouse is over us'
        if DraggableVertex.lock is not self:
            return
        if event.inaxes != self.circle.axes: return
        x0, y0, xpress, ypress = self.press
        dx = event.xdata - xpress
        dy = event.ydata - ypress
        self.circle.center = (x0 + dx, y0 + dy)
        self.parent.update()

    def on_release(self, event):
        'on release we reset the press data'
        if DraggableVertex.lock is not self:
            return
        self.press = None
        DraggableVertex.lock = None
        self.parent.update()

    def disconnect(self):
        'disconnect all the stored connection ids'
        self.canvas.mpl_disconnect(self.cid_press)
        self.canvas.mpl_disconnect(self.cid_release)
        self.canvas.mpl_disconnect(self.cid_motion)

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

        self.gate = None
        self.canvas = ax.figure.canvas
        self.ax = ax
        self.vertices = []
        self.poly = None
        self.background = None
        self.t = time.time()
        self.double_click_t = 1.0

        self.cid_press = self.canvas.mpl_connect(
            'button_press_event', self.onclick)
        self.cid_draw = self.canvas.mpl_connect(
            'draw_event', self.update_background)

    def add_vertex(self, vertex):
        # print vertex.center
        self.ax.add_patch(vertex)
        dv = DraggableVertex(vertex, self)
        dv.connect()
        self.vertices.append(dv)
        self.update()

    def update_background(self, event):
        self.background = self.canvas.copy_from_bbox(self.ax.bbox)

    def update(self):
        # print "updating"

        if len(self.vertices) >= 3:
            xy = numpy.array([v.circle.center for v in self.vertices])
            # bug in matplotlib? patch is not closed without this
            xy = numpy.concatenate([xy, [xy[0]]])
            if self.poly is None:
                self.poly = Polygon(xy, closed=True, alpha=0.5,
                                    facecolor='pink')
                self.ax.add_patch(self.poly)
            else:
                self.poly.set_xy(xy)

        if self.background is not None:
            self.canvas.restore_region(self.background)
        if self.poly is not None:
            self.ax.draw_artist(self.poly)
        for vertex in self.vertices:
            self.ax.draw_artist(vertex.circle)

        # print ">>>", self.ax.bbox

        self.canvas.blit(self.ax.bbox)

    def onclick(self, event):
        xmin, xmax, unused_ymin, unused_ymax = self.ax.axis()
        w = xmax - xmin

        if event.button == 3:
            vertex = Circle((event.xdata, event.ydata), radius=0.01 * w)
            self.add_vertex(vertex)

        # double left click triggers gating
        xy = numpy.array([v.circle.center for v in self.vertices])
        xypoints = numpy.array([[event.xdata, event.ydata]])

        if self.poly:
            if (event.button == 1 and
                points_inside_poly(xypoints, xy)):
                if (time.time() - self.t < self.double_click_t):
                    self.zoom_to_gate(event)
                self.t = time.time()

    def zoom_to_gate(self, event):
        xy = numpy.array([v.circle.center for v in self.vertices])
        gate = g(xy, self.idxs)
        gate.gate(self.fcm)
        self.gate = gate
        self.vertices = []
        self.poly = None
        self.ax.patches = []

        # get rid of old points
        del self.ax.collections[0]
        plt.close()
        
    def disconnect(self):
        'disconnect all the stored connection ids'
        self.canvas.mpl_disconnect(self.cid_press)
        self.canvas.mpl_disconnect(self.cid_draw)

def poly_gate(fcm, idxs):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    gate = Gate(fcm, idxs, ax)
    plt.show()
    return gate.gate

if __name__ == '__main__':
    import sys
    sys.path.append('../')
    from fcm import FCSreader

    fcm = FCSreader('../../sample_data/3FITC_4PE_004.fcs').get_FCMdata()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    idxs = [2, 3]
    gate = Gate(fcm, idxs, ax)

    plt.show()
    print fcm.tree.nodes
    print gate.gate
