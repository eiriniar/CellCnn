import os, sys
import numpy as np
import cPickle as pickle
import pandas as pd
from  sklearn.preprocessing import StandardScaler
from collections import Counter

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import fcluster

from lasagne import layers, init
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
import theano
import theano.tensor as T
import lasagne.layers.helper as lh  

from lifelines.estimation import KaplanMeierFitter
from lifelines.statistics import logrank_test
import scipy.stats as ss
from sklearn.cross_validation import KFold
import cellCnn
from cellCnn.utils import mkdir_p, param_vector
from cellCnn.run_CellCnn import train_model
from cellCnn.plotting import plot_marker_distribution
from numpy.random import RandomState
from lasagne.random import set_rng as set_lasagne_rng


WDIR = os.path.join(cellCnn.__path__[0], 'examples')
OUTDIR = os.path.join(WDIR, 'output', 'HIV')
mkdir_p(OUTDIR)

def plot_KM(stime, censor, g1, pval, figname):
    sns.set_style('white')
    kmf = KaplanMeierFitter()        
    f, ax = plt.subplots(figsize=(3, 3))
    np.set_printoptions(precision=2, suppress=False)
    kmf.fit(stime[g1], event_observed=censor[g1], label=["high-risk group"])
    kmf.plot(ax=ax, ci_show=False, show_censors=True)
    kmf.fit(stime[~g1], event_observed=censor[~g1], label=["low-risk group"])
    kmf.plot(ax=ax, ci_show=False, show_censors=True)
    ax.grid(b=False)
    sns.despine()
    plt.ylim(0,1)
    plt.xlabel("time", fontsize=14)
    plt.ylabel("survival", fontsize=14)
    plt.text(0.7, 0.85, 'pval = %.2e' % (pval), fontdict={'size': 12},
            horizontalalignment='center', verticalalignment='center',
            transform=ax.transAxes) 
    plt.xticks(rotation=45)
    for item in (ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(10)
    plt.tight_layout()
    plt.savefig(figname, format='eps')
    plt.close()

def logrank_pval(stime, censor, g1):
    res = logrank_test(stime[g1], stime[~g1], censor[g1], censor[~g1], alpha=.95)
    return res.p_value


def main():
       
    seed = 12345
    np.random.seed(seed)
    set_lasagne_rng(RandomState(seed))
     
    LOOKUP_PATH = os.path.join(WDIR, 'data', 'HIV.pkl')
    lookup =  pickle.load(open(LOOKUP_PATH, 'rb'))
    data_list = lookup['data']
    y = lookup['y']
    labels = lookup['labels']
    nmark = len(labels)
    
    # event occurence list    
    occurred = [x for i, x in enumerate(data_list) if y[i,1] == 1]
    not_occurred = [x for i, x in enumerate(data_list) if y[i,1] == 0]
    y1 = y[y[:,1] == 1]
    y0 = y[y[:,1] == 0]
    
    # split the examples randomly into a training (2/3) and test (1/3) cohort
    # both cohorts should contain equal percentage of cencored data
    sep1 = len(y1) / 3
    sep0 = len(y0) / 3
        
    # include only uncensored data from the training cohort for training CellCnn
    tr_list = occurred[sep1:]
    tr_stime = y1[sep1:,0].astype(float)
            
    # transform survival times to [-1, 1] interval by ranking them
    tr_stime = (ss.rankdata(tr_stime) / (0.5 * len(tr_stime))) - 1
                
    # fit scaler to all training data
    sc = StandardScaler()
    sc.fit(np.vstack(occurred[sep1:] + not_occurred[sep0:]))
    tr_list = [sc.transform(x) for x in tr_list]
            
    # the test cohort
    validation_list = [sc.transform(x) for x in (occurred[:sep1] + not_occurred[:sep0])]
    y_valid = np.vstack([y1[:sep1], y0[:sep0]])
    
    # cross validation on the training cohort    
    nfold = 10
    nfilter = 3
           
    skf = KFold(len(tr_list), n_folds=nfold, shuffle=True)
    committee = []
    valid_accuracy = []
    accum_w = np.empty((nfilter * nfold, nmark+2))
    
    for ifold, (train_index, test_index) in enumerate(skf):
        cv_train_samples = [tr_list[t_idx] for t_idx in train_index]
        cv_test_samples = [tr_list[t_idx] for t_idx in test_index]
        cv_y_train = list(tr_stime[train_index])
        cv_y_test = list(tr_stime[test_index])
        
        results = train_model(cv_train_samples, cv_y_train, labels,
                                valid_samples=cv_test_samples, valid_phenotypes=cv_y_test, 
                                ncell=500, nsubset=200, subset_selection='random',
                                nrun=3, pooling='mean', regression=True, nfilter=nfilter,
                                learning_rate=0.03, momentum=0.9, l2_weight_decay_conv=1e-8,
                                l2_weight_decay_out=1e-8, max_epochs=20, verbose=1,
                                select_filters='best', accur_thres=-1)
            
        net_dict = results['best_net']
            
        # update the committee of networks        
        committee.append(net_dict)
        valid_accuracy.append(results['best_accuracy'])
        w_tot = param_vector(net_dict, regression=True)
                
        # add weights to accumulator    
        accum_w[ifold*nfilter:(ifold+1)*nfilter] = w_tot
         
    save_path = os.path.join(OUTDIR, 'network_committee.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump((committee, valid_accuracy), f, -1)    
        
    '''
    committee, valid_accuracy = pickle.load(open(save_path, 'r'))    
    # retrieve the filter weights
    for ifold, net_dict in enumerate(committee):
        w_tot = param_vector(net_dict, regression=True)
                
        # add weights to accumulator    
        accum_w[ifold*nfilter:(ifold+1)*nfilter] = w_tot
    '''    
    
    # choose the strong signatures (all of them)
    w_strong = accum_w
    
    # members of each cluster should have cosine similarity > 0.7 
    # equivalently, cosine distance < 0.3
    Z = linkage(w_strong, 'average', metric='cosine')
    clusters = fcluster(Z, .3, criterion='distance') - 1
        
    n_clusters = len(np.unique(clusters))
    print '%d clusters chosen' % (n_clusters)   
            
    # plot the discovered filter profiles
    plt.figure(figsize=(3,2))
    idx = range(nmark) + [nmark+1]
    clmap = sns.clustermap(pd.DataFrame(w_strong[:,idx], columns=labels+['survival']),
                                method='average', metric='cosine', row_linkage=Z,
                                col_cluster=False, robust=True, yticklabels=clusters)
    clmap.cax.set_visible(False)
    fig_path = os.path.join(OUTDIR, 'HIV_clmap.eps')
    clmap.savefig(fig_path, format='eps')
    plt.close()
        
        
    # generate the consensus filter profiles
    c = Counter(clusters)
    cons = []
    for key, val in c.items():
        if val > nfold/2:
            cons.append(np.mean(w_strong[clusters == key], axis=0))
    cons_mat = np.vstack(cons)
        
    # plot the consensus filter profiles
    plt.figure(figsize=(10, 3))
    idx = range(nmark) + [nmark+1]
    ax = sns.heatmap(pd.DataFrame(cons_mat[:,idx], columns=labels + ['survival']),
                            robust=True, yticklabels=False)
    plt.xticks(rotation=90)
    ax.tick_params(axis='both', which='major', labelsize=20)
    plt.tight_layout()
    fig_path = os.path.join(OUTDIR, 'clmap_consensus.eps')
    plt.savefig(fig_path, format='eps')
    plt.close()
       
    # create an ensemble of neural networks
    ncell_cons = 3000
    ncell_voter = 3000
    layers_voter = [
                    (layers.InputLayer, {'name': 'input', 'shape': (None, nmark, ncell_voter)}),
                    (layers.Conv1DLayer, {'name': 'conv', 
                                        'num_filters': nfilter, 'filter_size': 1}),
                    (layers.Pool1DLayer, {'name': 'meanPool', 'pool_size' : ncell_voter,
                                        'mode': 'average_exc_pad'}),
                    (layers.DenseLayer, {'name': 'output',
                                        'num_units': 1,
                                        'nonlinearity': T.tanh})]
             
    # predict on the test cohort
    small_data_list_v = [x[:ncell_cons].T.reshape(1,nmark,ncell_cons) for x in validation_list]
    data_v = np.vstack(small_data_list_v)
    stime, censor = y_valid[:,0], y_valid[:,1]
    
    # committee of the best nfold/2 models
    voter_risk_pred = list()
    for ifold in np.argsort(valid_accuracy):
        voter = NeuralNet(layers = layers_voter,                
                                    update = nesterov_momentum,
                                    update_learning_rate = 0.001,
                                    regression=True,
                                    max_epochs=5,
                                    verbose=0)
        voter.load_params_from(committee[ifold])
        voter.initialize()
        # rank the risk predictions
        voter_risk_pred.append(ss.rankdata(- np.squeeze(voter.predict(data_v))))
    all_voters = np.vstack(voter_risk_pred)
                
    # compute mean rank per individual
    risk_p = np.mean(all_voters, axis=0)
    g1 = np.squeeze(risk_p > np.median(risk_p))
    voters_pval_v = logrank_pval(stime, censor, g1)
    fig_v = os.path.join(OUTDIR, 'committee_cox_test.eps')
    plot_KM(stime, censor, g1, voters_pval_v, fig_v) 
                  
    # filter-activating cells
    data_t = np.vstack(small_data_list_v)
    data_stack = np.vstack([x for x in np.swapaxes(data_t, 2, 1)])
                
    # finally define a network from the consensus filters
    nfilter_cons = cons_mat.shape[0]
    ncell_cons = 3000
    layers_cons = [
                    (layers.InputLayer, {'name': 'input', 'shape': (None, nmark, ncell_cons)}),
                    (layers.Conv1DLayer, {'name': 'conv', 
                                        'b': init.Constant(cons_mat[:,-2]),
                                        'W': cons_mat[:,:-2].reshape(nfilter_cons, nmark, 1),
                                        'num_filters': nfilter_cons, 'filter_size': 1}),
                    (layers.Pool1DLayer, {'name': 'meanPool', 'pool_size' : ncell_cons,
                                        'mode': 'average_exc_pad'}),
                    (layers.DenseLayer, {'name': 'output',
                                        'num_units': 1,
                                        'W': np.sign(cons_mat[:,-1:]),
                                        'b': init.Constant(0.),
                                        'nonlinearity': T.tanh})]
            
    net_cons = NeuralNet(layers = layers_cons,                
                            update = nesterov_momentum,
                            update_learning_rate = 0.001,
                            regression=True,
                            max_epochs=5,
                            verbose=0)
    net_cons.initialize()

    # get the representation after mean pooling
    xs = T.tensor3('xs').astype(theano.config.floatX)
    act_conv = theano.function([xs], lh.get_output(net_cons.layers_['conv'], xs)) 
    
    # and apply to the test data
    act_tot = act_conv(data_t)
    act_tot = np.swapaxes(act_tot, 2, 1)
    act_stack = np.vstack([x for x in act_tot])
    idx = range(7) + [8,9]
                
    for i_map in range(nfilter_cons):
        val = act_stack[:, i_map]
        descending_order = np.argsort(val)[::-1]
        val_cumsum = np.cumsum(val[descending_order])
        data_sorted = data_stack[descending_order]
        thres = 0.75 * val_cumsum[-1]
        res_data = data_sorted[val_cumsum < thres] 
        fig_path = os.path.join(OUTDIR, 'filter_'+str(i_map)+'_active.eps')       
        plot_marker_distribution([res_data[:,idx], data_stack[:,idx]],
                                            ['filter '+str(i_map), 'all'],
                                            [labels[l] for l in idx],
                                            (3,3), fig_path, 24)

   
if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.stderr.write("User interrupt!\n")
        sys.exit(-1)    