import sys
# sys.path.append("/Users/cliburn/MyPython") 

from fcm import loadFCS
from fcm.statistics import DPMixtureModel
from pylab import scatter, show, subplot, savefig
from fcm.graphics.plot import heatmap

import time

if __name__ == '__main__':
    #load data
    data = loadFCS('../sample_data/3FITC_4PE_004.fcs')
    heatmap(data, [(0,1),(0,2),(0,3),(2,3)], 2, 2, s=1, edgecolors='none',
            savefile='foo.tif')
    show()
