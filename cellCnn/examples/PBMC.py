import os, sys
import numpy as np
import cPickle as pickle
from sklearn import preprocessing
from sklearn.utils import shuffle

import cellCnn
from cellCnn.utils import mkdir_p
from cellCnn.run_CellCnn import train_model
from cellCnn.plotting import visualize_results
from numpy.random import RandomState
from lasagne.random import set_rng as set_lasagne_rng


WDIR = os.path.join(cellCnn.__path__[0], 'examples')
OUTDIR = os.path.join(WDIR, 'output', 'PBMC')
mkdir_p(OUTDIR)

def main():
    
    # set random seed for reproducible results
    seed = 12345
    np.random.seed(seed)
    set_lasagne_rng(RandomState(seed))

    # load the data (mass cytometry measurements after arcsinh transformation)
    LOOKUP_PATH = os.path.join(WDIR, 'data', 'GM-CSF_vs_control.pkl')
    lookup =  pickle.load(open(LOOKUP_PATH, 'rb'))
    labels = lookup[lookup.keys()[0]]['labels']
    
    # encode the cell type information that comes from manual gating
    # this information is not used by CellCnn
    # it's used only for validating the results 
    ctype = lookup[lookup.keys()[0]]['ctype']
    le = preprocessing.LabelEncoder() 
    le.fit(ctype)  
        
    # put together the data from different mass cytometry runs    
    accum_stimulated, accum_unstimulated = [], []
    accum_z_unstim, accum_z_stim = [], []
    for key, val in lookup.items():
        x = val['X']
        y = val['y']
        z = le.transform(val['ctype'])
        accum_stimulated.append(x[y == 1])
        accum_unstimulated.append(x[y == 0])
        accum_z_stim.append(z[y == 1])
        accum_z_unstim.append(z[y == 0])
        
    
    x_stim = np.vstack(accum_stimulated)
    x_unstim = np.vstack(accum_unstimulated)
    z_stim = np.hstack(accum_z_stim)
    z_unstim = np.hstack(accum_z_unstim)
    x_stim, z_stim = shuffle(x_stim, z_stim)
    x_unstim, z_unstim = shuffle(x_unstim, z_unstim)
    
    # in "train_samples" we store all information about single cell profiles
    # that is used by CellCnn - here only the intracellular markers are used
    train_samples = [x_unstim[:,10:], x_stim[:,10:]]
        
    # in "train_sample_flags" we store all information about single cell profiles
    # that is not actually used by CellCnn, but we want to use later
    # for validation of our results
    train_sample_flags = [np.hstack([x_unstim[:,:10], z_unstim.reshape(-1,1)]),
                          np.hstack([x_stim[:,:10], z_stim.reshape(-1,1)])]
        
    # the considered phenotypes
    train_phenotypes = [0, 1]
        
    # labels for the markers used by CellCnn
    phospho_markers = labels[10:]
        
    results = train_model(train_samples=train_samples, train_phenotypes=train_phenotypes,
                          labels=phospho_markers, train_sample_flags=train_sample_flags,
                          ncell=200, nsubset=4096, subset_selection='random',
                          pooling='max', ncell_pooled=3, nfilter=2,
                          select_filters='consensus', accur_thres=.99)
                        
    visualize_results(results, OUTDIR, prefix='GM-CSF')
                            
if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.stderr.write("User interrupt!\n")
        sys.exit(-1)