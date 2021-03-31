import numpy as np
from scipy.special import gamma

class Poisson:
    
    def __init__(self, name, num_clusters, data, rates=None):
        
        self.name = name

        self.type = 'poisson'
        
        self.c = num_clusters

        self.n_iter = 0

        # Can be a single parameter or a vector of parameters, usually the latter
        if not isinstance(rates, np.ndarray):
            # If none are given generate a vector of rate normally distributed around the sample mean with sample variance
            self.params_init = np.abs(np.random.normal(loc=np.mean(data), scale=np.sqrt(np.var(data)), size=self.c))
            self.params = self.params_init
            
        else:
            self.params_init = rates
            self.params = rates

        # Observation of the poisson random variable, should be a length n column vector where n is the number of observations
        self.data = data.reshape((len(data), 1)).astype(float)
        # Beware of 0 in the data, as if all zeros are clustered together, the algorithm will break
        # Define a value that's almost zero to compensate
        if not np.all(self.data):
            zero_replace = 1e-20
            zeros_idx = np.where(self.data == 0)[0]
            self.data[zeros_idx] = zero_replace

        # Compute likelihood and log likelihood
        self.update()
    


    def get_likelihood(self, obs):
        # obs must be an integer or a column vector
        return ( self.params**obs * np.exp( -1 * self.params) ) / gamma(obs + 1)


    def get_log_likelihood(self, obs):
        # obs must be an integer or a column vector
        return obs * np.log(self.params) - self.params - np.log(gamma(obs + 1))


    def maximise(self, q_h):
        self.params_old = self.params

        # Optimise the energy w.r.t to rate parameters, q_h is the optimmised variational distribution output from the expectation step
        self.params = np.sum(q_h * self.data, axis=0) / np.sum(q_h, axis=0)

        self.update()

        self.n_iter += 1


    def update(self):
         # Likelihood of each observation given the current rates
        self.likelihood = self.get_likelihood(self.data)

        # Log likelihood, up to proportionality, of each observation given the current rates
        self.log_likelihood = self.get_log_likelihood(self.data)



