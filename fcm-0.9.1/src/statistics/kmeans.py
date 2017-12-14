from scipy.cluster import vq
from modelresult import ModelResult

class KMeans(ModelResult):
    '''
    K means model
    '''

    def __init__(self, centroids):
        self.cent = centroids

    def classify(self, x):
        return vq.vq(x, self.centroids())[0]

    def mus(self):
        return self.cent

    def centroids(self):
        return self.mus()

