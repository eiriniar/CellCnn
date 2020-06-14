""" Copyright 2016-2017 ETH Zurich, Eirini Arvaniti and Manfred Claassen.

This module contains utility functions for defining a neural network.

"""

import numpy as np
from keras import backend as K


def float32(k):
    return np.cast['float32'](k)


def int32(k):
    return np.cast['int32'](k)


# this is the only time where Theano is directly used
def select_top(x, k):
    return K.mean(K.sort(x, axis=1)[:, -k:, :], axis=1)




