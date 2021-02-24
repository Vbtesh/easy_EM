import numpy as np
from methods.general import normArray

class Hidden:

    def __init__(self, num_clusters, init_dist='uniform'):
        self.name = 'clusters'

        self.c = num_clusters

        self.n_iter = 0
        
        # Start with a uniform distribution
        if init_dist == 'uniform':
            self.params_init = np.ones(num_clusters) / num_clusters
            self.params = np.ones(num_clusters) / num_clusters
        elif init_dist == 'random':
            params = np.random.uniform(size=self.c)
            self.params_init = params / params.sum()
            self.params = params / params.sum()

        self.log_params = np.log(self.params)


    def expectation(self, obs_log_likelihoods):
        # Observed log likelihoods given current maximised parameters for all observed variables
        dataset_size = obs_log_likelihoods[0].shape

        log_q_h = self.log_params * np.ones(dataset_size)

        for ll in obs_log_likelihoods:
            log_q_h += ll

        # Do the max log transformation
        # Subtract the maximum log from each row
        log_q_h = log_q_h - np.amax(log_q_h, axis=0)
        # Exponentiate and normalise resulting variational distribution
        q_h = np.exp(log_q_h)
        self.q_h = normArray(q_h)
        pass


    def maximise(self, q_h):
        self.params_old = self.params

        # Optimise the energy w.r.t to rate parameters, q_h is the optimmised variational distribution output from the expectation step
        self.params = normArray(np.mean(q_h, axis=0), axis=0)

        self.log_params = np.log(self.params)

        self.n_iter += 1