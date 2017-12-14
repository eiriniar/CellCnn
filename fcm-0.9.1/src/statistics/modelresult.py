

class ModelResult(object):
    '''
    Base Class for fitted model results
    '''
    def classify(self,x):
        '''
        assign events
        '''
        raise NotImplementedError("ModelResults should be subclassed and classify should be overridden")