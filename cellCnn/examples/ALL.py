import os, sys
import numpy as np
import cPickle as pickle
from sklearn.utils import shuffle

import cellCnn
from cellCnn.utils import mkdir_p
from cellCnn.run_CellCnn import train_model
from cellCnn.plotting import visualize_results
from numpy.random import RandomState
from lasagne.random import set_rng as set_lasagne_rng


''' 
    ALL.pkl can be downloaded from 
    http://www.imsb.ethz.ch/research/claassen/Software/cellcnn.html
'''

WDIR = os.path.join(cellCnn.__path__[0], 'examples')
LOOKUP_PATH = os.path.join(WDIR, 'data', 'ALL.pkl')

OUTDIR = os.path.join(WDIR, 'output', 'ALL')
mkdir_p(OUTDIR)

def main():
    
    # set random seed for reproducible results
    seed = 12345
    np.random.seed(seed)
    set_lasagne_rng(RandomState(seed))

    lookup =  pickle.load(open(LOOKUP_PATH, 'rb'))
    labels = lookup['labels']
    x_control = lookup['control']
    x_healthy = lookup['healthy']
    x_ALL = shuffle(lookup['ALL'])
    
    # create a flag to track ALL cells later on
    z_control = np.zeros((x_control.shape[0], 1), dtype=int)
    z_healthy = np.zeros((x_healthy.shape[0], 1), dtype=int)
    z_ALL = np.ones((x_ALL.shape[0], 1), dtype=int)
    
    # create the mixed sample
    x_mixed = np.vstack([x_healthy, x_ALL])
    z_mixed = np.vstack([z_healthy, z_ALL])   
    x_mixed, z_mixed = shuffle(x_mixed, z_mixed)

    # in "train_samples" we store all information about single cell profiles
    # that is used by CellCnn
    train_samples = [x_control, x_mixed]
        
    # in "train_sample_flags" we store all information about single cell profiles
    # that is not actually used by CellCnn, but we want to use later
    # for validation of our results
    train_sample_flags = [z_control, z_mixed]
        
    # the considered phenotypes: 0 for control, 1 for the sample containing ALL
    train_phenotypes = [0, 1]
        
    results = train_model(train_samples, train_phenotypes, labels,
                            train_sample_flags=train_sample_flags,
                            ncell=1000, nsubset=4096, subset_selection='outlier',
                            landmark_norm=['CD47', 'CD49d'], nrun=20,
                            pooling='max', regression=False, nfilter=3,
                            learning_rate=0.03, momentum=0.9, l2_weight_decay_conv=1e-8,
                            l2_weight_decay_out=1e-8, max_epochs=20, verbose=1,
                            select_filters='consensus_priority', accur_thres=.99)
                        
    visualize_results(results, OUTDIR, prefix='example_ALL',
                        plots=['consensus', 'clustering_results'])
                            
if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.stderr.write("User interrupt!\n")
        sys.exit(-1)