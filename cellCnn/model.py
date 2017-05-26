
""" Copyright 2016-2017 ETH Zurich, Eirini Arvaniti and Manfred Claassen.

This module contains functions for performing a CellCnn analysis.

"""

import sys
import os
import copy
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

from cellCnn.utils import combine_samples, normalize_outliers_to_control
from cellCnn.utils import cluster_profiles, keras_param_vector
from cellCnn.utils import generate_subsets, generate_biased_subsets
from cellCnn.utils import get_filters_classification, get_filters_regression
from cellCnn.utils import mkdir_p
from cellCnn.theano_utils import select_top, float32, int32, activity_KL

from keras.layers import Input, Dense, Lambda, Activation, Dropout
from keras.layers.convolutional import Convolution1D
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l1l2
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils.np_utils import to_categorical


class CellCnn(object):

    """ Creates a CellCnn model.

    Args:
        - ncell :
            Number of cells per multi-cell input.
        - nsubset :
            Total number of multi-cell inputs that will be generated per class, if
            `per_sample` = `False`. Total number of multi-cell inputs that will be generated from
            each input sample, if `per_sample` = `True`.
        - per_sample :
            Whether the `nsubset` argument refers to each class or each input sample.
            For regression problems, it is automatically set to `True`.
        - subset_selection :
            Can be 'random' or 'outlier'. Generate multi-cell inputs uniformly at
            random or biased towards outliers. The latter option is only relevant for detection of
            extremely rare (frequency < 0.1%) cell populations.
        - maxpool_percentages :
            A list specifying candidate percentages of cells that will be max-pooled per
            filter. For instance, mean pooling corresponds to `maxpool_percentages` = `[100]`.
        - nfilter_choice :
            A list specifying candidate numbers of filters for the neural network.
        - scale :
            Whether to z-transform each feature (mean = 0, standard deviation = 1) prior to
            training.
        - quant_normed :
            Whether the input samples have already been pre-processed with quantile
            normalization. In this case, each feature is zero-centered by subtracting 0.5.
        - nrun :
            Number of neural network configurations to try (should be set >= 3).
        - regression :
            Set to `True` for a regression problem. Default is `False`, which corresponds
            to a classification setting.
        - learning_rate :
            Learning rate for the Adam optimization algorithm. If set to `None`,
            learning rates in the range [0.001, 0.01] will be tried out.
        - dropout :
            Whether to use dropout (at each epoch, set a neuron to zero with probability
            `dropout_p`). The default behavior 'auto' uses dropout when `nfilter` > 5.
        - dropout_p :
            The dropout probability.
        - coeff_l1 :
            Coefficient for L1 weight regularization.
        - coeff_l2 :
            Coefficient for L2 weight regularization.
        - coeff_activity :
            Coefficient for regularizing the activity at each filter.
        - max_epochs :
            Maximum number of iterations through the data.
        - patience :
            Number of epochs before early stopping (stops if the validation loss does not
            decrease anymore).
        - dendrogram_cutoff :
            Cutoff for hierarchical clustering of filter weights. Clustering is
            performed using cosine similarity, so the cutof should be in [0, 1]. A lower cutoff will
            generate more clusters.
        - accur_thres :
            Keep filters from models achieving at least this accuracy. If less than 3
            models pass the accuracy threshold, keep filters from the best 3 models.
    """

    def __init__(self, ncell=200, nsubset=1000, per_sample=False, subset_selection='random',
                 maxpool_percentages=[0.01, 1., 5., 20., 100.], scale=True, quant_normed=False,
                 nfilter_choice=range(3, 10), dropout='auto', dropout_p=.5,
                 coeff_l1=0, coeff_l2=0.0001, coeff_activity=0, learning_rate=None,
                 regression=False, max_epochs=20, patience=5, nrun=15, dendrogram_cutoff=0.4,
                 accur_thres=.95, verbose=1):

        # initialize model attributes
        self.scale = scale
        self.quant_normed = quant_normed
        self.nrun = nrun
        self.regression = regression
        self.ncell = ncell
        self.nsubset = nsubset
        self.per_sample = per_sample
        self.subset_selection = subset_selection
        self.maxpool_percentages = maxpool_percentages
        self.nfilter_choice = nfilter_choice
        self.learning_rate = learning_rate
        self.coeff_l1 = coeff_l1
        self.coeff_l2 = coeff_l2
        self.coeff_activity = coeff_activity
        self.dropout = dropout
        self.dropout_p = dropout_p
        self.max_epochs = max_epochs
        self.patience = patience
        self.dendrogram_cutoff = dendrogram_cutoff
        self.accur_thres = accur_thres
        self.verbose = verbose
        self.results = None

    def fit(self, train_samples, train_phenotypes, outdir, valid_samples=None,
            valid_phenotypes=None, generate_valid_set=True):

        """ Trains a CellCnn model.

        Args:
            - train_samples :
                List with input samples (e.g. cytometry samples) as numpy arrays.
            - train_phenotypes :
                List of phenotypes associated with the samples in `train_samples`.
            - outdir :
                Directory where output will be generated.
            - valid_samples :
                List with samples to be used as validation set while training the network.
            - valid_phenotypes :
                List of phenotypes associated with the samples in `valid_samples`.
            - generate_valid_set :
                If `valid_samples` is not provided, generate a validation set
                from the `train_samples`.

        Returns:
            A trained CellCnn model with the additional attribute `results`. The attribute `results`
            is a dictionary with the following entries:

            - clustering_result : clustered filter weights from all runs achieving \
                validation accuracy above the specified threshold `accur_thres`
            - selected_filters : a consensus filter matrix from the above clustering result
            - best_3_nets : the 3 best models (achieving highest validation accuracy)
            - best_net : the best model
            - w_best_net : filter and output weights of the best model
            - accuracies : list of validation accuracies achieved by different models
            - best_model_index : list index of the best model
            - config : list of neural network configurations used
            - scaler : a z-transform scaler object fitted to the training data
            - n_classes : number of output classes
        """

        res = train_model(train_samples, train_phenotypes, outdir,
                          valid_samples, valid_phenotypes, generate_valid_set,
                          scale=self.scale, nrun=self.nrun, regression=self.regression,
                          ncell=self.ncell, nsubset=self.nsubset, per_sample=self.per_sample,
                          subset_selection=self.subset_selection,
                          maxpool_percentages=self.maxpool_percentages,
                          nfilter_choice=self.nfilter_choice,
                          learning_rate=self.learning_rate,
                          coeff_l1=self.coeff_l1, coeff_l2=self.coeff_l2,
                          dropout=self.dropout, dropout_p=self.dropout_p,
                          coeff_activity=self.coeff_activity, max_epochs=self.max_epochs,
                          patience=self.patience, dendrogram_cutoff=self.dendrogram_cutoff,
                          accur_thres=self.accur_thres, verbose=self.verbose)
        self.results = res
        return self

    def predict(self, new_samples, ncell_per_sample=None):

        """ Makes predictions for new samples.

        Args:
            - new_samples :
                List with input samples (numpy arrays) for which predictions will be made.
            - ncell_per_sample :
                Size of the multi-cell inputs (only one multi-cell input is created
                per input sample). If set to None, the size of the multi-cell inputs equals the
                minimum size in `new_samples`.

        Returns:
            y_pred : Phenotype predictions for `new_samples`.
        """

        if ncell_per_sample is None:
            ncell_per_sample = np.min([x.shape[0] for x in new_samples])
        print 'Predictions based on multi-cell inputs containing %d cells.' % ncell_per_sample

        # z-transform the new samples if we did that for the training samples
        scaler = self.results['scaler']
        if scaler is not None:
            new_samples = copy.deepcopy(new_samples)
            new_samples = [scaler.transform(x) for x in new_samples]

        nmark = new_samples[0].shape[1]
        n_classes = self.results['n_classes']

        # get the configuration of the top 3 models
        accuracies = self.results['accuracies']
        sorted_idx = np.argsort(accuracies)[::-1][:3]
        config = self.results['config']

        y_pred = np.zeros((3, len(new_samples), n_classes))
        for i_enum, i in enumerate(sorted_idx):
            nfilter = config['nfilter'][i]
            maxpool_percentage = config['maxpool_percentage'][i]
            ncell_pooled = max(1, int(maxpool_percentage/100. * ncell_per_sample))

            # build the model architecture
            model = build_model(ncell_per_sample, nmark,
                                nfilter=nfilter, coeff_l1=0, coeff_l2=0, coeff_activity=0,
                                k=ncell_pooled, dropout=False, dropout_p=0,
                                regression=self.regression, n_classes=n_classes, lr=0.01)

            # and load the learned filter and output weights
            weights = self.results['best_3_nets'][i_enum]
            model.set_weights(weights)

            # select a random subset of `ncell_per_sample` and make predictions
            new_samples = [shuffle(x)[:ncell_per_sample].reshape(1, ncell_per_sample, nmark)
                           for x in new_samples]
            data_test = np.vstack(new_samples)
            y_pred[i_enum] = model.predict(data_test)
        return np.mean(y_pred, axis=0)


def train_model(train_samples, train_phenotypes, outdir,
                valid_samples=None, valid_phenotypes=None, generate_valid_set=True,
                scale=True, quant_normed=False, nrun=20, regression=False,
                ncell=200, nsubset=1000, per_sample=False, subset_selection='random',
                maxpool_percentages=[0.01, 1., 5., 20., 100.], nfilter_choice=range(3, 10),
                learning_rate=None, coeff_l1=0, coeff_l2=1e-4, dropout='auto', dropout_p=.5,
                coeff_activity=0, max_epochs=20, patience=5,
                dendrogram_cutoff=0.4, accur_thres=.95, verbose=1):

    """ Performs a CellCnn analysis """
    mkdir_p(outdir)

    if nrun < 3:
        print 'The nrun argument should be >= 3, setting it to 3.'
        nrun = 3

    # copy the list of samples so that they are not modified in place
    train_samples = copy.deepcopy(train_samples)
    if valid_samples is not None:
        valid_samples = copy.deepcopy(valid_samples)

    # normalize extreme values
    # we assume that 0 corresponds to the control class
    if subset_selection == 'outlier':
        ctrl_list = [train_samples[i] for i in np.where(np.array(train_phenotypes) == 0)[0]]
        test_list = [train_samples[i] for i in np.where(np.array(train_phenotypes) != 0)[0]]
        train_samples = normalize_outliers_to_control(ctrl_list, test_list)

        if valid_samples is not None:
            ctrl_list = [valid_samples[i] for i in np.where(np.array(valid_phenotypes) == 0)[0]]
            test_list = [valid_samples[i] for i in np.where(np.array(valid_phenotypes) != 0)[0]]
            valid_samples = normalize_outliers_to_control(ctrl_list, test_list)

    # merge all input samples (X_train, X_valid)
    # and generate an identifier for each of them (train_id, valid_id)
    if (valid_samples is None) and (not generate_valid_set):
        sample_ids = range(len(train_phenotypes))
        X_train, id_train = combine_samples(train_samples, sample_ids)

    elif (valid_samples is None) and generate_valid_set:
        sample_ids = range(len(train_phenotypes))
        X, sample_id = combine_samples(train_samples, sample_ids)
        valid_phenotypes = train_phenotypes

        # split into train-validation partitions
        eval_folds = 5
        #kf = StratifiedKFold(sample_id, eval_folds)
        #train_indices, valid_indices = next(iter(kf))
        kf = StratifiedKFold(n_splits=eval_folds)
        train_indices, valid_indices = next(kf.split(X, sample_id))
        X_train, id_train = X[train_indices], sample_id[train_indices]
        X_valid, id_valid = X[valid_indices], sample_id[valid_indices]

    else:
        sample_ids = range(len(train_phenotypes))
        X_train, id_train = combine_samples(train_samples, sample_ids)
        sample_ids = range(len(valid_phenotypes))
        X_valid, id_valid = combine_samples(valid_samples, sample_ids)

    if quant_normed:
        z_scaler = StandardScaler(with_mean=True, with_std=False)
        z_scaler.fit(0.5 * np.ones((1, X_train.shape[1])))
        X_train = z_scaler.transform(X_train)
    elif scale:
        z_scaler = StandardScaler(with_mean=True, with_std=True)
        z_scaler.fit(X_train)
        X_train = z_scaler.transform(X_train)
    else:
        z_scaler = None

    X_train, id_train = shuffle(X_train, id_train)
    train_phenotypes = np.asarray(train_phenotypes)

    # an array containing the phenotype for each single cell
    y_train = train_phenotypes[id_train]

    if (valid_samples is not None) or generate_valid_set:
        if scale:
            X_valid = z_scaler.transform(X_valid)

        X_valid, id_valid = shuffle(X_valid, id_valid)
        valid_phenotypes = np.asarray(valid_phenotypes)
        y_valid = valid_phenotypes[id_valid]

    # number of measured markers
    nmark = X_train.shape[1]

    # generate multi-cell inputs
    print 'Generating multi-cell inputs...'

    if subset_selection == 'outlier':
        # here we assume that class 0 is always the control class
        x_ctrl_train = X_train[y_train == 0]
        to_keep = int(0.1 * (X_train.shape[0] / len(train_phenotypes)))
        nsubset_ctrl = nsubset / np.sum(train_phenotypes == 0)

        # generate a fixed number of subsets per class
        nsubset_biased = [0]
        for pheno in range(1, len(np.unique(train_phenotypes))):
            nsubset_biased.append(nsubset / np.sum(train_phenotypes == pheno))

        X_tr, y_tr = generate_biased_subsets(X_train, train_phenotypes, id_train, x_ctrl_train,
                                             nsubset_ctrl, nsubset_biased, ncell, to_keep,
                                             id_ctrl=np.where(train_phenotypes == 0)[0],
                                             id_biased=np.where(train_phenotypes != 0)[0])
        # save those because it takes long to generate
        #np.save(os.path.join(outdir, 'X_tr.npy'), X_tr)
        #np.save(os.path.join(outdir, 'y_tr.npy'), y_tr)
        #X_tr = np.load(os.path.join(outdir, 'X_tr.npy'))
        #y_tr = np.load(os.path.join(outdir, 'y_tr.npy'))

        if (valid_samples is not None) or generate_valid_set:
            x_ctrl_valid = X_valid[y_valid == 0]
            nsubset_ctrl = nsubset / np.sum(valid_phenotypes == 0)

            # generate a fixed number of subsets per class
            nsubset_biased = [0]
            for pheno in range(1, len(np.unique(valid_phenotypes))):
                nsubset_biased.append(nsubset / np.sum(valid_phenotypes == pheno))

            to_keep = int(0.1 * (X_valid.shape[0] / len(valid_phenotypes)))
            X_v, y_v = generate_biased_subsets(X_valid, valid_phenotypes, id_valid, x_ctrl_valid,
                                               nsubset_ctrl, nsubset_biased, ncell, to_keep,
                                               id_ctrl=np.where(valid_phenotypes == 0)[0],
                                               id_biased=np.where(valid_phenotypes != 0)[0])
            # save those because it takes long to generate
            #np.save(os.path.join(outdir, 'X_v.npy'), X_v)
            #np.save(os.path.join(outdir, 'y_v.npy'), y_v)
            #X_v = np.load(os.path.join(outdir, 'X_v.npy'))
            #y_v = np.load(os.path.join(outdir, 'y_v.npy'))
        else:
            cut = X_tr.shape[0] / 5
            X_v = X_tr[:cut]
            y_v = y_tr[:cut]
            X_tr = X_tr[cut:]
            y_tr = y_tr[cut:]
    else:
        # generate 'nsubset' multi-cell inputs per input sample
        if per_sample:
            X_tr, y_tr = generate_subsets(X_train, train_phenotypes, id_train,
                                          nsubset, ncell, per_sample)
            if (valid_samples is not None) or generate_valid_set:
                X_v, y_v = generate_subsets(X_valid, valid_phenotypes, id_valid,
                                            nsubset, ncell, per_sample)
        # generate 'nsubset' multi-cell inputs per class
        else:
            nsubset_list = []
            for pheno in range(len(np.unique(train_phenotypes))):
                nsubset_list.append(nsubset / np.sum(train_phenotypes == pheno))
            X_tr, y_tr = generate_subsets(X_train, train_phenotypes, id_train,
                                          nsubset_list, ncell, per_sample)

            if (valid_samples is not None) or generate_valid_set:
                nsubset_list = []
                for pheno in range(len(np.unique(valid_phenotypes))):
                    nsubset_list.append(nsubset / np.sum(valid_phenotypes == pheno))
                X_v, y_v = generate_subsets(X_valid, valid_phenotypes, id_valid,
                                            nsubset_list, ncell, per_sample)
    print 'Done.'

    ## neural network configuration ##
    # batch size
    bs = 200

    # keras needs (nbatch, ncell, nmark)
    X_tr = np.swapaxes(X_tr, 2, 1)
    X_v = np.swapaxes(X_v, 2, 1)
    n_classes = 1

    if not regression:
        n_classes = len(np.unique(train_phenotypes))
        y_tr = to_categorical(y_tr, n_classes)
        y_v = to_categorical(y_v, n_classes)

    # train some neural networks with different parameter configurations
    accuracies = np.zeros(nrun)
    w_store = dict()
    config = dict()
    config['nfilter'] = []
    config['learning_rate'] = []
    config['maxpool_percentage'] = []
    lr = learning_rate

    for irun in range(nrun):
        if verbose:
            print 'training network: %d' % (irun + 1)
        if learning_rate is None:
            lr = 10 ** np.random.uniform(-3, -2)
            config['learning_rate'].append(lr)

        # choose number of filters for this run
        nfilter = np.random.choice(nfilter_choice)
        config['nfilter'].append(nfilter)
        print 'Number of filters: %d' % nfilter

        # choose number of cells pooled for this run
        mp = maxpool_percentages[irun % len(maxpool_percentages)]
        config['maxpool_percentage'].append(mp)
        k = max(1, int(mp/100. * ncell))
        print 'Cells pooled: %d' % k

        # build the neural network
        model = build_model(ncell, nmark, nfilter,
                            coeff_l1, coeff_l2, coeff_activity, k,
                            dropout, dropout_p, regression, n_classes, lr)

        filepath = os.path.join(outdir, 'nnet_run_%d.hdf5' % irun)
        try:
            if not regression:
                check = ModelCheckpoint(filepath, monitor='val_loss', save_best_only=True,
                                        mode='auto')
                earlyStopping = EarlyStopping(monitor='val_loss', patience=patience, mode='auto')
                model.fit(float32(X_tr), int32(y_tr),
                          nb_epoch=max_epochs, batch_size=bs, callbacks=[check, earlyStopping],
                          validation_data=(float32(X_v), int32(y_v)), verbose=verbose)
            else:
                check = ModelCheckpoint(filepath, monitor='val_loss', save_best_only=True,
                                        mode='auto')
                earlyStopping = EarlyStopping(monitor='val_loss', patience=patience, mode='auto')
                model.fit(float32(X_tr), float32(y_tr),
                          nb_epoch=max_epochs, batch_size=bs, callbacks=[check, earlyStopping],
                          validation_data=(float32(X_v), float32(y_v)), verbose=verbose)

            # load the model from the epoch with highest validation accuracy
            model.load_weights(filepath)

            if not regression:
                valid_metric = model.evaluate(float32(X_v), int32(y_v))[-1]
                print 'Best validation accuracy: %.2f' % valid_metric
                accuracies[irun] = valid_metric

            else:
                train_metric = model.evaluate(float32(X_tr), float32(y_tr), batch_size=bs)
                print 'Best train loss: %.2f' % train_metric
                valid_metric = model.evaluate(float32(X_v), float32(y_v), batch_size=bs)
                print 'Best validation loss: %.2f' % valid_metric
                accuracies[irun] = - valid_metric

            # extract the network parameters
            w_store[irun] = model.get_weights()

        except Exception as e:
            sys.stderr.write('An exception was raised during training the network.\n')
            sys.stderr.write(str(e) + '\n')

    # the top 3 performing networks
    model_sorted_idx = np.argsort(accuracies)[::-1][:3]
    best_3_nets = [w_store[i] for i in model_sorted_idx]
    best_net = best_3_nets[0]
    best_accuracy_idx = model_sorted_idx[0]

    # weights from the best-performing network
    w_best_net = keras_param_vector(best_net)

    # post-process the learned filters
    # cluster weights from all networks that achieved accuracy above the specified thershold
    w_cons, cluster_res = cluster_profiles(w_store, nmark, accuracies, accur_thres,
                                           dendrogram_cutoff=dendrogram_cutoff)
    results = {
        'clustering_result': cluster_res,
        'selected_filters': w_cons,
        'best_net': best_net,
        'best_3_nets': best_3_nets,
        'w_best_net': w_best_net,
        'accuracies': accuracies,
        'best_model_index': best_accuracy_idx,
        'config': config,
        'scaler': z_scaler,
        'n_classes' : n_classes
    }

    if (valid_samples is not None) and (w_cons is not None):
        maxpool_percentage = config['maxpool_percentage'][best_accuracy_idx]
        if regression:
            tau = get_filters_regression(w_cons, z_scaler, valid_samples, valid_phenotypes,
                                         maxpool_percentage)
            results['filter_tau'] = tau

        else:
            filter_diff = get_filters_classification(w_cons, z_scaler, valid_samples,
                                                     valid_phenotypes, maxpool_percentage)
            results['filter_diff'] = filter_diff
    return results

def build_model(ncell, nmark, nfilter, coeff_l1, coeff_l2, coeff_activity,
                k, dropout, dropout_p, regression, n_classes, lr=0.01):

    """ Builds the neural network architecture """

    # the input layer
    data_input = Input(shape=(ncell, nmark))

    # the filters
    conv = Convolution1D(nfilter, 1, activation='linear',
                         W_regularizer=l1l2(l1=coeff_l1, l2=coeff_l2),
                         activity_regularizer=activity_KL(l=coeff_activity, p=0.05),
                         name='conv1')(data_input)
    conv = Activation('relu')(conv)
    # the cell grouping part
    pooled = Lambda(select_top, output_shape=(nfilter,), arguments={'k':k})(conv)

    # possibly add dropout
    if dropout or ((dropout == 'auto') and (nfilter > 5)):
        pooled = Dropout(p=dropout_p)(pooled)

    # network prediction output
    if not regression:
        output = Dense(n_classes, activation='softmax',
                       W_regularizer=l1l2(l1=coeff_l1, l2=coeff_l2),
                       name='output')(pooled)
    else:
        output = Dense(1, activation='tanh', W_regularizer=l1l2(l1=coeff_l1, l2=coeff_l2),
                       name='output')(pooled)
    model = Model(input=data_input, output=output)

    if not regression:
        model.compile(optimizer=Adam(lr=lr),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
    else:
        model.compile(optimizer=Adam(lr=lr),
                      loss='mean_squared_error')
    return model
