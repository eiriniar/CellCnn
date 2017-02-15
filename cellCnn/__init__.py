
# according to this issue: https://github.com/fchollet/keras/issues/828
# setting the optimizer to `fast_compile` avoids NaNs when training with ReLU

import theano
theano.config.optimizer = 'fast_compile'
