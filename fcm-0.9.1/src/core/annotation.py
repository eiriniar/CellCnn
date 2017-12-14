"""FCM annotation and annotation sets for FCM data and files
"""

class Annotation(object):
    """
    Annotation object for storing metadata about FCM data
    """

    def __init__(self, annotations=None):
        """
        Annotation([annotations = {}])
        """
        if annotations == None:
            annotations = {}

        self.__dict__['_mydict'] = annotations

    def __getattr__(self, name):
        """
        allow usage of annotation.foo or annotation[foo] to return the
        intended value
        """
        if name in self._mydict.keys():
            self.__dict__[name] = self._mydict[name]
            return self._mydict[name]
        else:
            try:
                return self._mydict.__getattribute__(name)
            except:
                raise AttributeError("'%s' has no attribue '%s'" % (str(self.__class__), name))

    def __getstate__(self):
        return self._mydict

    def __setstate__(self, dict):
        self.__dict__['_mydict'] = dict
        for i in dict.keys():
            self.__dict__[i] = dict[i]

    def __setattr__(self, name, value):
        """
        allow usage of annotation.foo  = x or annotation[foo] =x to set the
        intendede value
        """
        #return setattr(self._mydict, name, value)
        Annotation.__getattribute__(self, '_mydict')[name] = value
        self.__dict__[name] = value

    def __setitem__(self, name, value):
        """
        allow usage of annotation.foo  = x or annotation[foo] =x to set the
        intended value
        """
        self._mydict[name] = value
        self.__dict__[name] = self._mydict[name]

    def __getitem__(self, name):
        """
        allow usage of annotation.foo or annotation[foo] to return the
        intended value
        """
        return self._mydict[name]

    def __repr__(self):
        return 'Annotation(' + self._mydict.__repr__() + ')'

    def __getstate__(self):
        return self.__dict__

    def __setstate(self, state):
        self.__dict__ = state

    def __getinitargs__(self):
        return (self._mydict,)

    def copy(self):
        """
        D.copy() -> a shallow copy of D
        """
        return Annotation(self._mydict.copy())
