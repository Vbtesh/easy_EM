import numpy as np

class Gaussian_mean:
    
    def __init__(self, name, num_clusters, data, variance=None, means=None):
        
        self.name = name

        self.c = num_clusters

        self.type = 'gaussian_mean'

        self.n_iter = 0

        if variance:
            self.std = np.sqrt(variance)
        else:
            # Default is standard error
            self.std = np.sqrt(np.var(data.flatten())) / np.sqrt(len(data.flatten()))

        # Can be a single parameter or a vector of parameters, usually the latter
        if not isinstance(means, np.ndarray):
            # If none are given generate a vector of rate normally distributed around the sample mean with sample variance
            self.params_init = np.random.normal(loc=np.mean(data), scale=self.std, size=self.c)
            self.params = self.params_init
            
        else:
            self.initial_params = means
            self.params = means

        # Observation of the normal random variable, should be a length n column vector where n is the number of observations
        self.data = data.reshape((len(data), 1))

        # Compute likelihood and log likelihood
        self.update()


    def get_likelihood(self, obs):
        # obs must be an integer or a column vector
        return 1 / np.sqrt(2 * np.pi * self.std**2) * np.exp(- (1/(2 * self.std**2)) * (obs - self.params)**2)


    def get_log_likelihood(self, obs):
        # obs must be an integer or a column vector
        return - 1 / (2 * self.std**2) * (obs - self.params)**2


    def maximise(self, q_h):
        self.params_old = self.params

        # Optimise the energy w.r.t to mean parameters, q_h is the optimmised variational distribution output from the expectation step
        self.params = np.sum(q_h * self.data, axis=0) / np.sum(q_h, axis=0)

        self.update()

        self.n_iter += 1


    def update(self):
         # Likelihood of each observation given the current rates
        self.likelihood = self.get_likelihood(self.data)

        # Log likelihood, up to proportionality, of each observation given the current rates
        self.log_likelihood = self.get_log_likelihood(self.data)



