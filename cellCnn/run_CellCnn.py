import sys, os
import numpy as np
from sklearn.cross_validation import StratifiedKFold
from  sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
import copy

from cellCnn.utils import combine_samples, landmark_normalization, normalize_outliers_to_control
from cellCnn.utils import compute_consensus_profiles, param_vector
from cellCnn.utils import generate_subsets, generate_biased_subsets
from cellCnn.theano_utils import float32, int32, EarlyStopping, MyNeuralNet
from cellCnn.theano_utils import weight_decay_objective, SelectCellLayer
from cellCnn.downsample import knn_dist, knn_dist_memory_optimized

import theano
import theano.tensor as T
from lasagne import layers, init
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import BatchIterator, TrainSplit, NeuralNet


def train_model(train_samples, train_phenotypes, labels,
                valid_samples=None, valid_phenotypes=None, generate_valid_set=True,
                train_sample_flags=None, valid_sample_flags=None, 
                landmark_norm=None, scale=True,
                ncell=500, nsubset=4096, subset_selection='random', nrun=10,
                pooling='max', ncell_pooled=None, regression=False, nfilter=2,
                learning_rate=0.03, momentum=0.9, l2_weight_decay_conv=1e-8,
                l2_weight_decay_out=1e-8, max_epochs=10, verbose=1,
                select_filters='consensus', accur_thres=.9, benchmark_scores=False):
    
    '''
    train_samples: list with input samples, e.g. cytometry samples
    train_phenotype: phenotype associated with the samples in train_samples
    labels: labels of measured markers in train_samples
    '''
    
    # copy the list of samples so that they are not modified in place
    train_samples = copy.deepcopy(train_samples)
    if valid_samples is not None:
        valid_samples = copy.deepcopy(valid_samples)
        
    # create dummy single-cell flags if not given
    if train_sample_flags is None:
        train_sample_flags = [np.zeros((x.shape[0],1), dtype=int) for x in train_samples]
    if (valid_samples is not None) and (valid_sample_flags is None):
        valid_sample_flags = [np.zeros((x.shape[0],1), dtype=int) for x in valid_samples]

    if landmark_norm is not None:
        idx_to_normalize = [labels.index(label) for label in landmark_norm]
        train_samples = landmark_normalization(train_samples, idx_to_normalize)
    
        if valid_samples is not None:
            valid_samples = landmark_normalization(valid_samples, idx_to_normalize)
               
    # normalize extreme values
    # we assume that 0 corresponds to the control class
    if subset_selection == 'outlier':
        ctrl_list = [train_samples[i] for i in np.where(np.array(train_phenotypes) == 0)[0]]
        test_list = [train_samples[i] for i in np.where(np.array(train_phenotypes) == 1)[0]]
        train_samples = normalize_outliers_to_control(ctrl_list, test_list)

        if valid_samples is not None:
            ctrl_list = [valid_samples[i] for i in np.where(np.array(valid_phenotypes) == 0)[0]]
            test_list = [valid_samples[i] for i in np.where(np.array(valid_phenotypes) == 1)[0]]
            valid_samples = normalize_outliers_to_control(ctrl_list, test_list)


    if (valid_samples is None) and (not generate_valid_set):
        sample_ids = range(len(train_phenotypes))
        X_train, id_train, z_train = combine_samples(train_samples, sample_ids, train_sample_flags)
        
    elif (valid_samples is None) and generate_valid_set:
        sample_ids = range(len(train_phenotypes))
        X, sample_id, z = combine_samples(train_samples, sample_ids, train_sample_flags)
        valid_phenotypes = train_phenotypes                        
        
        # split into train-validation partitions
        eval_folds = 5
        kf = StratifiedKFold(sample_id, eval_folds)
        train_indices, valid_indices = next(iter(kf))
        X_train, id_train, z_train = X[train_indices], sample_id[train_indices], z[train_indices]
        X_valid, id_valid , z_valid = X[valid_indices], sample_id[valid_indices], z[valid_indices]
    
    else:
       sample_ids = range(len(train_phenotypes))
       X_train, id_train, z_train = combine_samples(train_samples, sample_ids, train_sample_flags)
       sample_ids = range(len(valid_phenotypes))
       X_valid, id_valid, z_valid = combine_samples(valid_samples, sample_ids, valid_sample_flags)

    # scale all marker distributions to mu=0, std=1
    if scale:
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
    
    X_train, z_train, id_train = shuffle(X_train, z_train, id_train)
    train_phenotypes = np.asarray(train_phenotypes)
    y_train = train_phenotypes[id_train]

    if (valid_samples is not None) or generate_valid_set:
        if scale:
            X_valid = scaler.transform(X_valid)
        X_valid, z_valid, id_valid = shuffle(X_valid, z_valid, id_valid)
        valid_phenotypes = np.asarray(valid_phenotypes)
        y_valid = valid_phenotypes[id_valid]


    # number of measured markers
    nmark = X_train.shape[1]
   
    # generate multi-cell inputs
    if subset_selection == 'outlier':
        
        # here we assume that class 0 is always the control class and class 1 is the test class
        # TODO: extend for more classes
        x_ctrl_train = X_train[y_train == 0]
        nsubset_ctrl = nsubset / np.sum(train_phenotypes == 0)
        nsubset_biased = nsubset / np.sum(train_phenotypes == 1)
        to_keep = int(0.01 * (X_train.shape[0] - x_ctrl_train.shape[0]))
        
        X_tr, y_tr = generate_biased_subsets(X_train, train_phenotypes, id_train, x_ctrl_train,
                                            nsubset_ctrl, nsubset_biased, ncell, to_keep,
                                            id_ctrl=np.where(train_phenotypes == 0)[0],
                                            id_biased=np.where(train_phenotypes == 1)[0])
        
        if (valid_samples is not None) or generate_valid_set:
            x_ctrl_valid = X_valid[y_valid == 0]
            nsubset_ctrl = nsubset / np.sum(valid_phenotypes == 0)
            nsubset_biased = nsubset / np.sum(valid_phenotypes == 1)
            to_keep = int(0.01 * (X_valid.shape[0] - x_ctrl_valid.shape[0]))
            X_v, y_v = generate_biased_subsets(X_valid, valid_phenotypes, id_valid, x_ctrl_valid,
                                                nsubset_ctrl, nsubset_biased, ncell, to_keep,
                                                id_ctrl=np.where(valid_phenotypes == 0)[0],
                                                id_biased=np.where(valid_phenotypes == 1)[0])
                                            
    # TODO: right now equal number of subsets is drawn from each sample
    # Do it per phenotype instead? 
    elif subset_selection == 'kmeans':
        X_tr, y_tr = generate_subsets(X_train, train_phenotypes, id_train,
                                        nsubset, ncell, k_init=True)
        if (valid_samples is not None) or generate_valid_set:
            X_v, y_v = generate_subsets(X_valid, valid_phenotypes, id_valid,
                                        nsubset/2, ncell, k_init=True)
    else:
        X_tr, y_tr = generate_subsets(X_train, train_phenotypes, id_train,
                                        nsubset, ncell, k_init=False)
        if (valid_samples is not None) or generate_valid_set:
            X_v, y_v = generate_subsets(X_valid, valid_phenotypes, id_valid,
                                        nsubset/2, ncell, k_init=False)

    ## neural network configuration ##
    
    # batch size
    bs = 128
        
    # the input and convolutional layers
    input_conv_layers = [
            (layers.InputLayer, {'name': 'input', 'shape': (None, nmark, ncell)}),
            (layers.Conv1DLayer, {'name': 'conv', 
                                'b': init.Constant(0.),
                                'W': init.Uniform(range=0.01),
                                'num_filters': nfilter, 'filter_size': 1})]

    # the pooling layer
    # max-pooling detects cell presence
    # mean-pooling detects cell frequency
    if pooling == 'max':
        if ncell_pooled is None:
            pooling_layers = [(layers.MaxPool1DLayer, {'name': 'maxPool',
                                                        'pool_size' : ncell})]
        else:
            pooling_layers = [
                (SelectCellLayer, {'name': 'select',
                                   'num_cell': ncell_pooled}),
                (layers.Pool1DLayer, {'name': 'maxPool',
                                      'pool_size' : ncell_pooled,
                                      'mode': 'average_exc_pad'})]

    elif pooling == 'mean':
        pooling_layers = [(layers.Pool1DLayer, {'name': 'meanPool',
                                                  'pool_size' : ncell,
                                                  'mode': 'average_exc_pad'})]
    else:
        sys.stderr.write("Undefined pooling type: %s\n" % pooling)
        sys.exit(-1)


        
    # the output layer
    if not regression:
        n_out = len(np.unique(train_phenotypes))
        output_nonlinearity = T.nnet.softmax
    else:
        n_out = 1
        output_nonlinearity = T.tanh
        
    output_layers = [(layers.DenseLayer, {'name': 'output',
                                          'num_units': n_out,
                                          'W': init.Uniform(range=0.01),
                                          'b': init.Constant(0.),
                                          'nonlinearity': output_nonlinearity})]
    # combine all the network layers
    layers_0 =  input_conv_layers + pooling_layers + output_layers
     
    # train some neural networks with different parameter configurations
    w_store = dict() 
    accuracies = np.empty(nrun)
      
    for irun in range(nrun):
        
        if verbose:
            print 'training network: %d' % (irun + 1)

        if (valid_samples is not None) or generate_valid_set:
            # build a convolutional neural network                                                                        
            net1 = MyNeuralNet(
                    layers = layers_0,
                    
                    # objective function and weight decay penalties
                    objective = weight_decay_objective,
                    objective_penalty_conv = l2_weight_decay_conv,
                    objective_penalty_output = l2_weight_decay_out,
                                    
                    # optimization method
                    update = nesterov_momentum,
                    update_learning_rate = theano.shared(float32(learning_rate)),
                    update_momentum = theano.shared(float32(momentum)),
                                
                    # batches
                    batch_iterator_train = BatchIterator(batch_size = bs),
                    batch_iterator_test = BatchIterator(batch_size = bs),
                    on_epoch_finished = [EarlyStopping(patience=3)],
                    
                    train_split = TrainSplit(eval_size=None),
                    regression = regression,
                    max_epochs = max_epochs,
                    verbose=verbose)
            
            # train the model
            if regression:
                net1.fit(float32(X_tr), float32(y_tr.reshape(-1,1)),
                        float32(X_v), float32(y_v.reshape(-1,1)))    
                valid_loss = net1.score(float32(X_v), float32(y_v.reshape(-1,1)))
                valid_accuracy = - valid_loss
            else:
                net1.fit(float32(X_tr), int32(y_tr), float32(X_v), int32(y_v))    
                valid_accuracy = net1.score(float32(X_v), int32(y_v))
        
        else:    
            # build a convolutional neural network without validation set                                                                     
            net1 = NeuralNet(
                    layers = layers_0,
                    
                    # objective function and weight decay penalties
                    objective = weight_decay_objective,
                    objective_penalty_conv = l2_weight_decay_conv,
                    objective_penalty_output = l2_weight_decay_out,
                                    
                    # optimization method
                    update = nesterov_momentum,
                    update_learning_rate = theano.shared(float32(learning_rate)),
                    update_momentum = theano.shared(float32(momentum)),
                                
                    # batches
                    batch_iterator_train = BatchIterator(batch_size = bs),
                    batch_iterator_test = BatchIterator(batch_size = bs),
                    on_epoch_finished = [],
                    
                    train_split = TrainSplit(eval_size=None),
                    regression = regression,
                    max_epochs = max_epochs,
                    verbose=verbose)
            
            # train the model
            if regression:
                net1.fit(float32(X_tr), float32(y_tr.reshape(-1,1)))    
                valid_accuracy = 0
            else:
                net1.fit(float32(X_tr), int32(y_tr))    
                valid_accuracy = 0

    
        # extract the network parameters
        w_store[irun] = net1.get_all_params_values()
        accuracies[irun] = valid_accuracy
    
    # which filter weights should we return
    # 'best': return the filter weights of the model with highest validation accuracy
    # 'consensus': return consensus filters based on hierarchical clustering
    # 'consensus_priority': prioritize the consensus filter that corresponds 
    #                       to the biggest cluster

    # this option only makes sense if validation samples were provided/generated
    best_net, w_best_net, best_accuracy = None, None, None
    if select_filters == 'best':
        best_net = w_store[np.argmax(accuracies)]
        w_best_net = param_vector(best_net, regression)
        best_accuracy = np.max(accuracies)
        w_cons, cluster_res = compute_consensus_profiles(w_store, accuracies, accur_thres,
                                                    regression, prioritize=False)
    elif select_filters == 'consensus':
        w_cons, cluster_res = compute_consensus_profiles(w_store, accuracies, accur_thres,
                                                    regression, prioritize=False)
    elif select_filters == 'consensus_priority':
        w_cons, cluster_res = compute_consensus_profiles(w_store, accuracies, accur_thres,
                                                    regression, prioritize=True)
    else:
        sys.stderr.write("Undefined option for selecting filters: %s\n" % select_filters)
        sys.exit(-1)

        print 'undefined option for selecting filters'

    if (valid_samples is not None) or generate_valid_set:
        X = np.vstack([X_train, X_valid])
        y = np.hstack([y_train, y_valid])
        z = np.vstack([z_train, z_valid])
    else:
        X = X_train
        y = y_train
        z = z_train

    # predict using CellCnn
    if select_filters == 'consensus_priority':           
        params = w_cons
        w, b = params[:-2], params[-2]
        x1  = X[y == 1]
        x0 = X[y == 0]
        cnn_pred = np.sum(w.reshape(1,-1) * x1, axis=1) + b
    else:
        cnn_pred = None
    
    results = {
                'clustering_result': cluster_res,
                'best_net': best_net,
                'w_best_net': w_best_net,
                'selected_filters': w_cons,
                'accuracies': accuracies,
                'best_accuracy': best_accuracy,
                'cnn_pred': cnn_pred,
                'labels': labels,
                'X': X,
                'y': y,
                'z': z}
    
    if benchmark_scores:

        # predict using outlier detection
        outlier_pred = knn_dist_memory_optimized(x1, x0, s=200000)
        
        # predict using multi-cell input logistic regression
        X_tr_mean = np.sum(X_tr, axis=-1)
        clf = LogisticRegression(C=10000, penalty='l2')
        clf.fit(X_tr_mean, y_tr)
        w_lr, b_lr = clf.coef_, clf.intercept_
        mean_pred = np.sum(w_lr.reshape(1,-1) * x1, axis=1) + b_lr[0]
        
        # predict using single-cell input logistic regression
        clf_sc = LogisticRegression(C=10000, penalty='l2')
        clf_sc.fit(X, y)
        w_lr, b_lr = clf_sc.coef_, clf_sc.intercept_
        sc_pred = np.sum(w_lr.reshape(1,-1) * x1, axis=1) + b_lr[0]
        
        # store the predictions
        results['outlier_pred'] = outlier_pred
        results['mean_pred'] = mean_pred
        results['sc_pred'] = sc_pred

    return results


