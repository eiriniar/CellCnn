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

WDIR = os.path.join(cellCnn.__path__[0], 'examples')
OUTDIR = os.path.join(WDIR, 'output', 'AML')
mkdir_p(OUTDIR)

def main():
    
    # set random seed for reproducible results
    seed = 12345
    np.random.seed(seed)
    set_lasagne_rng(RandomState(seed))

    LOOKUP_PATH = os.path.join(WDIR, 'data', 'AML.pkl')
    lookup =  pickle.load(open(LOOKUP_PATH, 'rb'))
    labels = lookup['labels']
    healthy_BM = lookup['healthy_BM']
    control_list = [x for (key, x) in healthy_BM[:-1]]
    x_healthy = healthy_BM[-1][1]
    
    # how many AML blast cells to spike-in
    nblast_spike = 1000
    x_AML = shuffle(lookup['aml_0'])[:nblast_spike]
    
    # create a flag to track AML cells later on
    z_control_list = [np.zeros((x.shape[0], 1), dtype=int) for x in control_list]
    z_healthy = np.zeros((x_healthy.shape[0], 1), dtype=int)
    z_AML = np.ones((x_AML.shape[0], 1), dtype=int)
    
    # create the mixed sample
    x_mixed = np.vstack([x_healthy, x_AML])
    z_mixed = np.vstack([z_healthy, z_AML])   
    x_mixed, z_mixed = shuffle(x_mixed, z_mixed)

    # in "train_samples" we store all information about single cell profiles
    # that is used by CellCnn
    train_samples = [np.vstack(control_list), x_mixed]
        
    # in "train_sample_flags" we store all information about single cell profiles
    # that is not actually used by CellCnn, but we want to use later
    # for validation of our results
    train_sample_flags = [np.vstack(z_control_list), z_mixed]
        
    # the considered phenotypes: 0 for control, 1 for the sample containing AML
    train_phenotypes = [0, 1]
        
    results = train_model(train_samples, train_phenotypes, labels,
                            train_sample_flags=train_sample_flags,
                            ncell=1000, nsubset=4096, subset_selection='outlier',
                            landmark_norm=['CD44'], nrun=20,
                            pooling='max', regression=False, nfilter=3,
                            learning_rate=0.02, momentum=0.9, l2_weight_decay_conv=1e-8,
                            l2_weight_decay_out=1e-8, max_epochs=20, verbose=1,
                            select_filters='consensus_priority', accur_thres=.99)
                        
    visualize_results(results, OUTDIR, prefix='example_AML')

                            
if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.stderr.write("User interrupt!\n")
        sys.exit(-1)