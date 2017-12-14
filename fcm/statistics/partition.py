import numpy


# do we even use this?
class Partition(object):
    """Stores information about how a data set was partitioned.

    Typically returned by some clustering function."""

    def __init__(self, fcm, component, p=None, z=None):
        """
        fcm = fcmdata object
        component = component object
        p = n*k matrix indicating probablility of each event belonging to kth partition
        z = n*1 array indicator integer of which partition an event belongs to.
        Either p or z but not both can be None.
        """
        self.fcm = fcm
        self.component = component
        if p is None and z is None:
            raise(Warning("Neither p nor z arguments specified."))
        self.p = p
        if z is not None:
            self.z = z
        else:
            self.z = self._set_z(z)

    def _set_z(self, z):
        """Returns array of indicator values."""
        if z is None:
            try:
                self.z = numpy.argmax(self.p, 1)
            except:
                raise(Warning("Neither p nor z is defined"))
        return self.z

    def get_k(self, k, theta=None):
        """Returns index where z==k with events < theta probability of belonging to k rejected"""
        if theta is not None:
            assert(0 <= theta <= 1.0)
        if self.z is None:
            return numpy.array([])
        else:
            return numpy.nonzero((self.z == k) & numpy.any(self.p > theta, 1))[0]

if __name__ == '__main__':
    pass
