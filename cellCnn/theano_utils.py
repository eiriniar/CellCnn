
""" Copyright 2016-2017 ETH Zurich, Eirini Arvaniti and Manfred Claassen.

This module contains utility functions for defining a neural network.

"""

import numpy as np
import theano.tensor as T
from keras.layers import Layer
from keras import backend as K
from keras import regularizers
from keras.regularizers import Regularizer


def float32(k):
    return np.cast['float32'](k)

def int32(k):
    return np.cast['int32'](k)

# this is the only time where Theano is directly used
def select_top(x, k):
    return K.mean(T.sort(x, axis=1)[:, -k:, :], axis=1)

def kl_divergence(p, p_hat):
    return (p * K.log(p / p_hat)) + ((1-p) * K.log((1-p) / (1-p_hat)))

class KL_ActivityRegularizer(Regularizer):
    def __init__(self, l=0., p=0.1):
        self.l = K.cast_to_floatx(l)
        self.p = K.cast_to_floatx(p)
        self.uses_learning_phase = True

    def set_layer(self, layer):
        self.layer = layer

    def __call__(self, loss):
        if not hasattr(self, 'layer'):
            raise Exception('Need to call `set_layer` on '
                            'ActivityRegularizer instance '
                            'before calling the instance.')
        regularized_loss = loss
        for i in range(len(self.layer.inbound_nodes)):
            output = K.sigmoid(0.1 * self.layer.get_output_at(i))
            #output = self.layer.get_output_at(i)
            p_hat = K.mean(K.abs(output))
            regularized_loss += self.l * kl_divergence(self.p, p_hat)
        return K.in_train_phase(regularized_loss, loss)

    def get_config(self):
        return {'name': self.__class__.__name__,
                'l': float(self.l),
                'p': float(self.p)}

def activity_KL(l=0.01, p=0.1):
    return KL_ActivityRegularizer(l=l, p=p)

class ParametricSigmoid(Layer):
    def __init__(self, beta_init=0.1, weights=None, activity_regularizer=None, **kwargs):
        self.supports_masking = True
        self.beta_init = K.cast_to_floatx(beta_init)
        self.initial_weights = weights
        self.activity_regularizer = regularizers.get(activity_regularizer)
        super(ParametricSigmoid, self).__init__(**kwargs)

    def build(self, input_shape):
        input_shape = input_shape[1:]
        self.betas = K.variable(self.beta_init * np.ones(input_shape),
                                name='{}_betas'.format(self.name))
        self.trainable_weights = [self.betas]

        self.regularizers = []
        if self.activity_regularizer:
            self.activity_regularizer.set_layer(self)
            self.regularizers.append(self.activity_regularizer)

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def call(self, x, mask=None):
        return K.sigmoid(self.betas * x)

    def get_config(self):
        config = {'beta_init': self.beta_init,
                  'activity_regularizer': self.activity_regularizer.get_config()
                                          if self.activity_regularizer else None,}
        base_config = super(ParametricSigmoid, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
