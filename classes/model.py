import numpy as np
from classes.gaussian import Gaussian_mean
from classes.poisson import Poisson
from classes.multinomial import Multinomial
from classes.hidden import Hidden

class Model:

    def __init__(self, name, num_clusters, var_dict, data, df=True):

        self.name = name

        self.c = num_clusters

        self.n_iter = 0

        self.var_obs = np.array(list(var_dict.keys()))
        
        if df:
            self.data = data[self.var_obs]
        else:
            self.data = data[:, self.var_obs]

        # Var dict shape
        # Dict of dict
        # Keys are variable names (same as in the dataset)
        ## Secondary keys and values: 
        ### var_type: poisson, multinomial, gaussian_mean, etc
        ### params: None or vector/matrix of parameters appropriate for the distribution
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
            idx += 1

        # Initialise distribution over hidden variable
        # Default is uniform, can set it to 'random'
        self.hidden = Hidden(self.c, init_dist='uniform')


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

        self.clusters = []
        self.clusters_size = []

        for i in range(self.c):
            self.clusters.append(np.where(prob_assign == i)[0])
            self.clusters_size.append(len(self.clusters[i]))

        self.unclassified = np.where(prob_assign == self.c + 1)[0]
        self.unclassified_rate = len(self.unclassified) / len(prob_assign)

    
    def summary(self, sig=0):
        # Generate clusters:
        self.generate_clusters(sig=sig)

        # print summary
        print(f'Summary for {self.name} \n')

        print('Parameter estimates:')
        print(f'Hidden states: {self.c}')
        print('Distribution: ', self.hidden.params, '\n')
        
        for k in self.variables:
            print(f'Variable {k.name}')
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
        


            



