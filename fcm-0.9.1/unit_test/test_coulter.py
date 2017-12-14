"""Test reading of Coulter FCS files."""

from fcm.io import FCSreader
import pylab

if __name__ == '__main__':
    print "BD"
    bd_file = '/Users/cliburn/hg/fcm/sample_data/3FITC_4PE_004.fcs'
    bd = FCSreader(bd_file)
    bd_data = bd.get_FCMdata()
    print bd_data.shape

    print "Coulter"
    coulter_file = '/Users/cliburn/hg/fcm/sample_data/coulter.fcs'
    coulter = FCSreader(coulter_file)
    coulter_data = coulter.get_FCMdata()
    print coulter_data.shape
#     pylab.subplot(2,2,1)
#     pylab.scatter(xs[:,0], xs[:,1], s=1)
#     pylab.subplot(2,2,2)
#     pylab.scatter(xs[:,2], xs[:,3], s=1)
#     pylab.subplot(2,2,3)
#     pylab.scatter(xs[:,4], xs[:,5], s=1)
#     pylab.subplot(2,2,4)
#     pylab.scatter(xs[:,6], xs[:,7], s=1)
    
#     pylab.savefig('coulter.png')
