import numpy as np
import theano.tensor as T
from nolearn.lasagne import NeuralNet
from lasagne.layers import Layer, get_output
from lasagne.objectives import aggregate
from lasagne.regularization import regularize_layer_params, l2
from sklearn.preprocessing import LabelEncoder
from time import time


def float32(k):
    return np.cast['float32'](k)
    
def float64(k):
    return np.cast['float64'](k)    
    
def int32(k):
    return np.cast['int32'](k) 
    
def relu(x):
    return T.switch(x<0, 0, x)


class SelectCellLayer(Layer):
    '''
    Defines a custom layer for selecting cells
    with top cell filter activity.
    '''

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


def weight_decay_objective(layers,
                        loss_function,
                        target,
                        penalty_conv=1e-8,
                        penalty_conv_type = l2,
                        penalty_output=1e-8,
                        penalty_output_type = l2,
                        aggregate=aggregate,
                        deterministic=False,
                        get_output_kw={}):
    '''
    Defines L2 weight decay on network weights. 
    '''
    net_out = get_output(layers[-1], deterministic=deterministic,
                        **get_output_kw)
    loss = loss_function(net_out, target)
    p1 = penalty_conv * regularize_layer_params(layers[1], penalty_conv_type)
    p2 = penalty_output * regularize_layer_params(layers[-1], penalty_output_type)
    losses = loss + p1 + p2
    return aggregate(losses)

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
    '''
     Slightly adapts the NeuralNet class defined in nolearn,
     so that it accepts a predefined validation set.
    '''
        
    def fit(self, X_train, y_train, X_valid, y_valid, epochs=None):
        X_train, y_train = self._check_good_input(X_train, y_train)
        X_valid, y_valid = self._check_good_input(X_valid, y_valid)

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
