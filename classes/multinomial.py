import numpy as np
from methods.general import normArray

class Multinomial:

    def __init__(self, name, num_clusters, data, num_outcomes, probs=None):
        self.name = name

        self.c = num_clusters

        self.type = 'multinomial'

        self.n_iter = 0

        if num_outcomes:
            self.o = num_outcomes
        else:
            self.o = len(np.unique(data))

        # Probs should be a vector of probabilities
        # If not given, initialise randomly
        if not isinstance(probs, np.ndarray):
            probs = np.random.uniform(size=(self.o, self.c))
            self.params_init = normArray(probs, axis=0)
            self.params = normArray(probs, axis=0)
        else:
            self.params_init = probs 
            self.params = probs
            
        # Observation of the multinomial random variable, should be a length n column vector where n is the number of observations
        self.data = data.flatten().reshape((len(data), 1))

        # Compute likelihood and log likelihood
        self.update()


    def get_likelihood(self, obs):
        # obs must be an integer or a column vector
        if isinstance(obs, np.ndarray):
            return self.params[obs.flatten().astype(int), :]
        else:
            return self.params[obs, :]


    def get_log_likelihood(self, obs):
        # obs must be an integer or a column vector
        if isinstance(obs, np.ndarray):
            return np.log(self.params[obs.flatten().astype(int), :])
        else:
            return np.log(self.params[obs, :])


    def maximise(self, q_h):
        self.params_old = self.params

        # Optimise the energy w.r.t to probs parameters, q_h is the optimmised variational distribution output from the expectation step
        params = np.zeros(self.params.shape)

        for i in np.arange(self.data.size):
            params[self.data[i].astype(int)] += q_h[i]

        self.params = normArray(params, axis=0)

        self.update()

        self.n_iter += 1


    def update(self):
         # Likelihood of each observation given the current rates
        self.likelihood = self.get_likelihood(self.data)

        # Log likelihood, up to proportionality, of each observation given the current rates
        self.log_likelihood = self.get_log_likelihood(self.data)