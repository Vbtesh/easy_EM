import numpy as np
from copy import deepcopy
from classes.gaussian import Gaussian_mean
from classes.poisson import Poisson
from classes.multinomial import Multinomial
from classes.hidden import Hidden
from methods.general import normArray

class Model:

    def __init__(self, name, num_clusters, var_dict, data, df=True, hidden_init='uniform'):

        self.name = name

        self.c = num_clusters

        self.n_iter = 0

        self.var_obs = np.array(list(var_dict.keys()))
        
        if df:
            self.data = data[self.var_obs]
            self.n = len(self.data)
        else:
            self.data = data[:, self.var_obs]
            self.n = self.data.shape[0]

        # Initialise distribution over hidden variable
        # Default is uniform, can set it to 'random'
        self.hidden = Hidden(self.c, init_dist=hidden_init)

        # Compute from all variables, the number of parameters to estimate
        self.num_params = len(self.hidden.params)

        # Var dict shape
        # Dict of dict
        # Keys are variable names (same as in the dataset)
        ## Secondary keys and values: 
        ### var_type: poisson, multinomial, gaussian_mean, etc
        ### params: None or vector/matrix of parameters appropriate for the distribution
        ### add_params: Additional parameters when applicable: num outcomes for multinomial, variance for gaussian_mean
        self.var_names = {}
        idx = 0
        self.variables = []
        for k, v in var_dict.items():

            # If df is given, turn data series into numpy array
            if df:
                loc_data = self.data[k].to_numpy()
            else:
                loc_data = self.data[:, k]
                
            if v['type'] == 'multinomial':
                self.variables.append(Multinomial(k, self.c, loc_data, v['add_params'], probs=v['params']))
            elif v['type'] == 'poisson':
                self.variables.append(Poisson(k, self.c, loc_data, rates=v['params']))
            elif v['type'] == 'gaussian_mean':
                self.variables.append(Gaussian_mean(k, self.c, loc_data, v['add_params'], means=v['params']))

            self.var_names[idx] = k

            self.num_params += self.variables[idx].params.size

            idx += 1
        
        # Compute degrees of freedom 
        self.df = self.n - self.num_params
        

    def run_EM(self, n_iter):

        for _ in np.arange(n_iter):
            
            # Update list of log_likelihoods
            log_likelihoods = []
            for k in self.variables:
                log_likelihoods.append(k.log_likelihood)

            # Expectation step
            self.hidden.expectation(log_likelihoods)

            # Maximisation steps
            self.hidden.maximise(self.hidden.q_h)
            for k in self.variables:
                k.maximise(self.hidden.q_h)

            self.n_iter += 1


    def generate_clusters(self, sig=0):
        # Splits the sample in cluster based on:
        ## The posterior distribution q(h)
        ## The significance level (sig=0 implies the argmax of the posterior)
        self.sig = sig
        prob_assign = np.argmax(self.hidden.q_h, axis=1)
        prob_max = np.amax(self.hidden.q_h, axis=1)

        tsh = np.where(prob_max < self.sig)[0]

        prob_assign[tsh] = self.c + 1

        self.assignments = prob_assign

        self.clusters = []
        self.clusters_size = []

        for i in range(self.c):
            self.clusters.append(np.where(prob_assign == i)[0])
            self.clusters_size.append(len(self.clusters[i]))

        self.unclassified = np.where(prob_assign == self.c + 1)[0]
        self.unclassified_rate = len(self.unclassified) / len(prob_assign)

    
    # Compute the marginal likelihood and log likelihood of the data given the learned distribution over hidden states
    def compute_model_metrics(self, sig=0):
        # Builds a posterior where all probs in q_h higher than sig are evaluated to ones
        # This is to avoid the risk of underflow
        self.post_q = deepcopy(self.hidden.q_h)

        self.post_q[np.where(self.post_q > sig)] = 1
        self.post_q[np.where(self.post_q < 1-sig)] = 0
        self.post_q = normArray(self.post_q)


        # Compute log likelihood by using the assignment a imputed value
        hidden_likelihood = np.sum(self.post_q * self.hidden.params, axis=1)

        sample_likelihood = hidden_likelihood

        for k in self.variables:
            sample_likelihood *= np.sum(self.post_q * k.likelihood, axis=1)

        # Sum up log likelihood for each data point.
        self.data_log_likelihood_ew = np.log(sample_likelihood)
        self.data_log_likelihood = np.sum(self.data_log_likelihood_ew)

        # Get negative log likelihood
        self.neg_ll = - self.data_log_likelihood

        # Compute log likelihood per df for model comparison
        if self.df:
            self.ll_per_df = self.data_log_likelihood / self.df
        else:
            self.ll_per_df = self.data_log_likelihood

        # Compute BIC for model comparison
        self.bic = self.num_params * np.log(self.n) - 2 * self.data_log_likelihood

        # Compute AIC for model comparison
        self.aic = self.num_params * 2 - 2 * self.data_log_likelihood

        # Corrected aic
        self.aicc = self.aic + (2 * self.num_params**2 + 2*self.num_params) / (self.n - self.num_params - 1)

        # Compile model comparison metrics
        self.model_metrics = {
            'log_likelihood': self.data_log_likelihood,
            'log_likelihood_per_df': self.ll_per_df,
            'neg_ll': self.neg_ll,
            'bic': self.bic,
            'aic': self.aic,
            'aicc': self.aicc
        }
        

    # Summarises the current state of the model after learning
    def summary(self, sig=0):
        # Generate clusters:
        self.generate_clusters(sig=sig)

        # print summary
        print(f'Summary for {self.name} \n')
        print(f'N = {self.n}, number of parameters: {self.num_params}')
        print(f'Degrees of freedom: {self.df}')
        print(f'Negative Log likelihood: {self.neg_ll}')
        print(f'BIC: {self.bic}')
        print(f'AIC: {self.aic}')

        print('Initial latent parameters:', self.hidden.params_init, '\n')

        print('Parameter estimates:')
        print(f'Hidden states: {self.c}')

        print('Final distribution: ', self.hidden.params, '\n')
        
        for k in self.variables:
            print(f'Variable: {k.name}')
            print(f'Type: {k.type}')
            print('Parameters:')
            for i in range(self.c):
                if k.type == 'multinomial':
                    print(f'Cluster {i}', k.params[:, i])
                else:
                    print(f'Cluster {i}', k.params[i])

            print()
                    

        print('Cluster summary')
        print('Sizes')
        for i in range(self.c):
            print(f'Cluster {i}:', self.clusters_size[i])
        print('\nUnclassified:', self.unclassified, 'Rate:', self.unclassified_rate)
        


            



