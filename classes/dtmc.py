import numpy as np
from methods.general import normArray

class dtmc():
    
    def __init__(self, name, num_clusters, data, states_mapping=None, t_matrix=None):
        self.name = name

        self.states = np.unique(data)
        
        # Generate random transition probability matrix if none is provided
        if t_matrix:
            self.params = t_matrix
        else:
            params = np.random.uniform(size=(num_clusters,len(self.states)))
            self.params = normArray(params)

        # Data should be a np array of n x t where n is the number of datapoints and t is the number of states observed
        # States should be labelled 0, 1, ..., t, if not indexing will fail.
        self.data = data



