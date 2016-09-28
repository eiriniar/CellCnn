import numpy as np
import theano.tensor as T
#from nolearn.lasagne import NeuralNet
#from lasagne import layers, init
#from lasagne.layers import Layer, Conv1DLayer, get_output
#from lasagne.objectives import aggregate, categorical_crossentropy, squared_error
#from lasagne.regularization import regularize_network_params, regularize_layer_params, l2, l1
from sklearn.preprocessing import LabelEncoder
from time import time

from keras.layers import Dense, Layer
from keras import backend as K
from keras import regularizers
from keras.regularizers import Regularizer


def float32(k):
    return np.cast['float32'](k)
    
def float64(k):
    return np.cast['float64'](k)    
    
def int32(k):
    return np.cast['int32'](k) 
    
def relu(x):
    return T.switch(x<0, 0, x)

def select_top(x, k):
    return T.mean(T.sort(x, axis=1)[:,-k:,:], axis=1)

def select_top_robust(x, k):
    return T.mean(T.sort(x, axis=1)[:,-(k+20):-20,:], axis=1)

def select_thres(x, thres=.6):
    max_feat = thres * T.max(x, axis=(0,1))
    max_feat = T.reshape(max_feat, (1,1,-1))
    max_feat = T.tile(max_feat, reps=(x.shape[0], x.shape[1], 1))
    x = x * (x > max_feat)
    return T.sum(x, axis=1) / 100

def multi_class_acc(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = (K.flatten(y_pred) > 0.5)
    return T.mean(T.eq(y_true_f, y_pred_f))

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
                  'activity_regularizer': self.activity_regularizer.get_config() if self.activity_regularizer else None,}
        base_config = super(ParametricSigmoid, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ForwardLayer(Dense):
    def __init__(self, **kwargs):
        super(Forward, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = input_shape[1]
        initial_weight_value = np.random.random(output_dim)
        self.W = K.variable(initial_weight_value)
        self.trainable_weights = [self.W]

    def call(self, x, mask=None):
        output = K.dot(x, T.linalg.diag(T.nlinalg.diag(self.W)))
        if self.bias:
            output += self.b
        return self.activation(output)

'''
class SelectCellLayer(Layer):
 
    def __init__(self, incoming, num_cell=5, to_keep='high', **kwargs):
        super(SelectCellLayer, self).__init__(incoming, **kwargs)
        self.num_cell = num_cell
        self.to_keep = to_keep

    def get_output_shape_for(self, input_shape):
        return [input_shape[0], input_shape[1], self.num_cell]
        
    def get_output_for(self, input, **kwargs):
        if self.to_keep == 'low':        
            return T.sort(input, axis=-1)[:,:,:self.num_cell]
        else:
            return T.sort(input, axis=-1)[:,:,-self.num_cell:]


class CustomMaxPoolLayer(Layer):

    def __init__(self, incoming, pool_function=T.argmax, **kwargs):
        super(CustomMaxPoolLayer, self).__init__(incoming, **kwargs)
        self.pool_function = pool_function

    def get_output_shape_for(self, input_shape):
        return [input_shape[0], input_shape[1], input_shape[1]]
        
    def get_output_for(self, input, **kwargs):
        
        assert input.ndim == 3
        nobs, nfeat, ncell = input.shape
        
        fat_x = input.dimshuffle(1, 0, 2).reshape((nfeat, -1), ndim=2)
        ind = self.pool_function(input, axis=2)
        offset = T.arange(nobs) * ncell
        ind = ind + offset.dimshuffle(0, 'x') 
        s_ind = ind.flatten()

        output = fat_x[:,s_ind]
        return output.dimshuffle(1,0).flatten().reshape((nobs,nfeat,nfeat),ndim=3).dimshuffle(0,2,1)


class GaussConv1DLayer(Conv1DLayer):

    def __init__(self, incoming, num_filters, filter_size, W=init.GlorotUniform(),
                 b=init.Constant(0.1), nonlinearity=None, **kwargs):
        super(GaussConv1DLayer, self).__init__(incoming, num_filters=num_filters,
                                                filter_size=filter_size,
                                                W=W, b=b, nonlinearity=nonlinearity,
                                                flip_filters=False, **kwargs)

    # TODO: b should be (nfilter, nmark)

    def convolve(self, input, **kwargs):
        w = self.W.dimshuffle('x',0,1,2)
        w = T.addbroadcast(w, 3)
        b = self.b.dimshuffle('x',0,'x')
        diff = w - input.dimshuffle(0,'x',1,2)
        diff_norm = T.sum(diff ** 2, axis=2)
        return T.exp(-diff_norm * (b ** 2))


# code adapted from pylearn2, sparse autoencoders
def KL(x, p, sigm_scale, sum_axis=(0, 2)):
    p_hat = T.mean(T.nnet.sigmoid(sigm_scale * x), axis=sum_axis)
    #kl = p * T.log(p / p_hat) + (1 - p) * \
    #        T.log((1 - p) / (1 - p_hat))
    return T.sum(p_hat)

def mixed_loss(net_out, ae_out, target):
    loss = categorical_crossentropy(net_out, target[:,0,0]) + 0.1 * squared_error(ae_out, target[:,:,1:])
    return loss


def weight_decay_objective(layers,
                        loss_function,
                        target,
                        coeff_l1=1e-8,
                        coeff_l2=1e-8,
                        coeff_KL=0,
                        target_KL=0.05,
                        sigmoid_scale_KL=1,
                        aggregate=aggregate,
                        deterministic=False,
                        get_output_kw={}):
    
    ae_out, net_out = get_output([layers[2], layers[-1]], deterministic=deterministic,
                        **get_output_kw)
    loss = loss_function(net_out, ae_out, target)
    p1 = coeff_l1 * regularize_network_params(layers[-1], l1)
    p2 = coeff_l2 * regularize_network_params(layers[-1], l2)
    #p1 = coeff_l1 * regularize_layer_params(layers[-1], l1)
    #p3 = 1e-4 * regularize_layer_params(layers[1], l1)
    #p2 = coeff_l2 * regularize_layer_params(layers[1], l2)
    losses = loss + p1 + p2
    
    if coeff_KL != 0:
        mean_conv_activation = get_output(layers[1], deterministic=deterministic,
                                            **get_output_kw)
        p_KL = coeff_KL * KL(mean_conv_activation, target_KL, sigmoid_scale_KL)
        losses = losses + p_KL

    return aggregate(losses)


def sparse_ae_objective(layers,
                        loss_function,
                        target,
                        coeff=0.00001,
                        p=0.05,
                        aggregate=aggregate,
                        deterministic=False,
                        get_output_kw={}):
    
    net_out = get_output(layers[-1], deterministic=deterministic,
                        **get_output_kw)
    loss = loss_function(net_out, target)
    
    if coeff != 0:
        mean_hidden_activation = get_output(layers[1], deterministic=deterministic,
                                            **get_output_kw)
        p_KL = coeff * KL(mean_hidden_activation, p, 1, sum_axis=0)
        loss = loss + p_KL

    return aggregate(loss)


# from Lasagne Recipes
class ModifiedBackprop(object):

    def __init__(self, nonlinearity):
        self.nonlinearity = nonlinearity
        self.ops = {}  # memoizes an OpFromGraph instance per tensor type

    def __call__(self, x):
        # We note the tensor type of the input variable to the nonlinearity
        # (mainly dimensionality and dtype); we need to create a fitting Op.
        tensor_type = x.type
        # If we did not create a suitable Op yet, this is the time to do so.
        if tensor_type not in self.ops:
            # For the graph, we create an input variable of the correct type:
            inp = tensor_type()
            # We pass it through the nonlinearity (and move to GPU if needed).
            outp = self.nonlinearity(inp)
            # Then we fix the forward expression...
            op = theano.OpFromGraph([inp], [outp])
            # ...and replace the gradient with our own (defined in a subclass).
            op.grad = self.grad
            # Finally, we memoize the new Op
            self.ops[tensor_type] = op
        # And apply the memoized Op to the input we got.
        return self.ops[tensor_type](x)

def compile_saliency_function(net):
    """
    Compiles a function to compute the saliency maps and predicted classes
    for a given minibatch of input images.
    """
    inp = net['input'].input_var
    outp = lasagne.layers.get_output(net['fc1'], deterministic=True)

    # the winner class
    max_outp = T.max(outp, axis=1)
    saliency = theano.grad(max_outp.sum(), wrt=inp)
    max_class = T.argmax(outp, axis=1)
    return theano.function([inp], [saliency, max_class])


# taken from the nolearn tutorial
# http://danielnouri.org/notes/2014/12/17/using-convolutional-neural-nets-to-detect-facial-keypoints-tutorial/

class AdjustVariable(object):
    def __init__(self, name, start=0.03, stop=0.001):
        self.name = name
        self.start, self.stop = start, stop
        self.ls = None

    def __call__(self, nn, train_history):
        if self.ls is None:
            self.ls = np.linspace(self.start, self.stop, nn.max_epochs)

        epoch = train_history[-1]['epoch']
        new_value = float32(self.ls[epoch - 1])
        getattr(nn, self.name).set_value(new_value)                                                                                              

# taken from the nolearn tutorial
# http://danielnouri.org/notes/2014/12/17/using-convolutional-neural-nets-to-detect-facial-keypoints-tutorial/

class EarlyStopping(object):
    def __init__(self, patience=100):
        self.patience = patience
        self.best_valid = np.inf
        self.best_valid_epoch = 0
        self.best_weights = None

    def __call__(self, nn, train_history):
        current_valid = train_history[-1]['valid_loss']
        current_epoch = train_history[-1]['epoch']
        if current_valid < self.best_valid:
            self.best_valid = current_valid
            self.best_valid_epoch = current_epoch
            self.best_weights = nn.get_all_params_values()
        elif self.best_valid_epoch + self.patience < current_epoch:
            print("Early stopping.")
            print("Best valid loss was {:.6f} at epoch {}.".format(
                self.best_valid, self.best_valid_epoch))
            nn.load_params_from(self.best_weights)
            raise StopIteration() 


class MyNeuralNet(NeuralNet):
    
     # Slightly adapts the NeuralNet class defined in nolearn,
     # so that it accepts a predefined validation set.
    
        
    def fit(self, X_train, y_train, X_valid, y_valid, epochs=None):
        #X_train, y_train = self._check_good_input(X_train, y_train)
        #X_valid, y_valid = self._check_good_input(X_valid, y_valid)

        if self.use_label_encoder:
            self.enc_ = LabelEncoder()
            y_train = self.enc_.transform(y_train).astype(np.int32)
            y_valid = self.enc_.transform(y_valid).astype(np.int32)            
            self.classes_ = self.enc_.classes_
        self.initialize()

        try:
            self.train_loop(X_train, y_train, X_valid, y_valid, epochs=epochs)
        except KeyboardInterrupt:
            pass
        return self

    def partial_fit(self, X_train, y_train, X_valid, y_valid, classes=None):
        return self.fit(X_train, y_train, X_valid, y_valid, epochs=1)

    def train_loop(self, X_train, y_train, X_valid, y_valid, epochs=None):
        epochs = epochs or self.max_epochs

        on_batch_finished = self.on_batch_finished
        if not isinstance(on_batch_finished, (list, tuple)):
            on_batch_finished = [on_batch_finished]

        on_epoch_finished = self.on_epoch_finished
        if not isinstance(on_epoch_finished, (list, tuple)):
            on_epoch_finished = [on_epoch_finished]

        on_training_started = self.on_training_started
        if not isinstance(on_training_started, (list, tuple)):
            on_training_started = [on_training_started]

        on_training_finished = self.on_training_finished
        if not isinstance(on_training_finished, (list, tuple)):
            on_training_finished = [on_training_finished]

        epoch = 0
        best_valid_loss = (
            min([row['valid_loss'] for row in self.train_history_]) if
            self.train_history_ else np.inf
            )
        best_train_loss = (
            min([row['train_loss'] for row in self.train_history_]) if
            self.train_history_ else np.inf
            )
        for func in on_training_started:
            func(self, self.train_history_)

        num_epochs_past = len(self.train_history_)
        learned_w = []

        while epoch < epochs:
            epoch += 1

            train_losses = []
            valid_losses = []
            valid_accuracies = []
            if self.custom_scores:
                custom_scores = [[] for _ in self.custom_scores]
            else:
                custom_scores = []

            t0 = time()


            for Xb, yb in self.batch_iterator_train(X_train, y_train):
                batch_train_loss = self.apply_batch_func(
                    self.train_iter_, Xb, yb)
                train_losses.append(batch_train_loss)

                # NOT tested
                # get the current filter weights
                curr_params = self.get_all_params_values()
                #learned_w.append(curr_params['conv'][0].flatten())

                for func in on_batch_finished:
                    func(self, self.train_history_)

            for Xb, yb in self.batch_iterator_test(X_valid, y_valid):
                batch_valid_loss, accuracy = self.apply_batch_func(
                    self.eval_iter_, Xb, yb)
                valid_losses.append(batch_valid_loss)
                valid_accuracies.append(accuracy)

                if self.custom_scores:
                    y_prob = self.apply_batch_func(self.predict_iter_, Xb)
                    for custom_scorer, custom_score in zip(self.custom_scores, custom_scores):
                        custom_score.append(custom_scorer[1](yb, y_prob))

            avg_train_loss = np.mean(train_losses)
            avg_valid_loss = np.mean(valid_losses)
            avg_valid_accuracy = np.mean(valid_accuracies)
            if custom_scores:
                avg_custom_scores = np.mean(custom_scores, axis=1)

            if avg_train_loss < best_train_loss:
                best_train_loss = avg_train_loss
            if avg_valid_loss < best_valid_loss:
                best_valid_loss = avg_valid_loss

            info = {
                'epoch': num_epochs_past + epoch,
                'train_loss': avg_train_loss,
                'train_loss_best': best_train_loss == avg_train_loss,
                'valid_loss': avg_valid_loss,
                'valid_loss_best': best_valid_loss == avg_valid_loss,
                'valid_accuracy': avg_valid_accuracy,
                'dur': time() - t0,
                #'learned_w': learned_w,
                }
            if self.custom_scores:
                for index, custom_score in enumerate(self.custom_scores):
                    info[custom_score[0]] = avg_custom_scores[index]
            self.train_history_.append(info)

            try:
                for func in on_epoch_finished:
                    func(self, self.train_history_)
            except StopIteration:
                break

        for func in on_training_finished:
            func(self, self.train_history_) 
'''