import numpy as np
from classes.model import Model

# An object with a set of methods to generate and train multiple models and compare clusters
# Uses monte carlo like methods to approximately explore the optima of the marginal distributions of observed variables given the latent variable
# Generate a given number of models, trains them starting from different initial conditions (define how to do this exactly) and then outputs the different results and their frequencies

class Collection:

    def __init__(self, name, num_models, var_dict, data, num_clusters, df=True, hidden_init='uniform'):
        self.name = name

        # Can be an int or a np.ndarray of integers with all number of clusters to explore
        self.c = num_clusters

        self.trained = False

        self.multiple_init = False
        # The initial distribution for the latent variable can either be random, uniform or defined
        self.hidden_init = hidden_init
        self.N = num_models
        # If it is defined, there is 2 possibilities:
        ## if hidden_init is a 1 x self.c array : then use it for all models
        ## else if it is a k x self.c array : use each distribution in the array as many times as possible given num_models
        if isinstance(hidden_init, np.ndarray):
            if hidden_init.shape[0] > 1:
                self.multiple_init = True
                num_series = num_models % hidden_init.shape[0]
                self.hidden_init = np.tile(hidden_init, num_series)
                self.N = self.hidden_init.shape[0]


        self.multiple_c = False
        # Set up models
        if self.multiple_init:
            self.models = [Model(f'{self.name}_{n}', self.c, var_dict, data=data, df=df, hidden_init=self.hidden_init[n,:]) for n in np.arange(self.N)]
            
        else:
            if isinstance(self.c, list):
                self.models = []
                for cluster in self.c:
                    self.models.append([Model(f'{self.name}_{n}', cluster, var_dict, data=data, df=df, hidden_init=self.hidden_init) for n in np.arange(self.N)])
                self.multiple_c = True
            else:
                self.models = [Model(f'{self.name}_{n}', self.c, var_dict, data=data, df=df, hidden_init=self.hidden_init) for n in np.arange(self.N)]


    # Trains all models in self.models
    def train_models(self, n_iter, sig=0):
        
         
        print(f'Training {self.N} models with {self.c} clusters...')

        if self.multiple_c:
            for i, clusters in enumerate(self.c):
                for n in np.arange(self.N):
                    self.models[i][n].run_EM(n_iter=n_iter)
                    if n % 10 == 0:
                        print(f'c={clusters}, model {n}...')
        else:
            for n in np.arange(self.N):
                self.models[n].run_EM(n_iter=n_iter)
                if n % 10 == 0:
                    print(f'{n}...')

            print(f'{n} done.')
        
        
        self.trained = True
        

    # Finds all possible optima and their frequencies
    # Decimals define the rounding for the parameters
    # Can only be called outside of compute_mle if there is only one case of clusters
    def aggregate_optima(self, models, sig=0, decimals=10):
        # Return void if models have not been trained
        if not self.trained:
            print('Cannot aggregate infos from untrained models, use Collection.train_models() before calling this method.')
            return

        posteriors = []
        posteriors_freq = []
        inits = []
        log_likelihoods = []
        ll_per_df = []
        instance_models = []
        
        count_nans = 0
        # Find all unique posteriors, frequencies and initial distributions
        for n in np.arange(self.N):
            model = models[n]
            model.compute_model_metrics(sig=sig)

            if True in np.isnan(model.hidden.params):
                count_nans += 1
                continue 

            post_sorted = list(np.sort(np.around(model.hidden.params, decimals), axis=None))

            if post_sorted not in posteriors:

                posteriors.append(post_sorted)
                inits.append([model.hidden.params_init])
                posteriors_freq.append(1)
                log_likelihoods.append(model.data_log_likelihood)
                ll_per_df.append(model.ll_per_df)
                instance_models.append(model)

            else:
                idx = posteriors.index(post_sorted)
                inits[idx].append(model.hidden.params_init)
                posteriors_freq[idx] += 1
            
        freq_nans = count_nans / self.N


        posteriors = np.array(posteriors)
        posteriors_freq = np.array(posteriors_freq)

        # Find mle model
        mle_idx = np.argmax(log_likelihoods)
        mle_model = instance_models[mle_idx]
        

        # Compile output
        optima_dict = {
            'num_clusters': models[0].c, 
            'posterior': posteriors, 
            'posterior_freq': posteriors_freq, 
            'data_log_likelihood': log_likelihoods, 
            'll_per_df': ll_per_df, 
            'model_instances': instance_models,
            'freq_nan_models': freq_nans
            }

        return mle_model, optima_dict


    # Finds the model with the highest log ll to degree of freedom ratio
    # If summary is true, prints a summary for the mle model
    def maximum_likelihood_estimation(self, sig, comparison_metric, summary=True, assignment=True, decimals=10):

        self.comparison_metric = comparison_metric

        if not self.multiple_c:
            self.mle_model, self.mle_analysis_dict = self.aggregate_optima(models=self.models, sig=sig, decimals=decimals)

        else:
            self.mle_analysis = []
            self.metrics = []
            self.mle_model = self.models[0][0]

            for i, cluster in enumerate(self.c):
                mle_model, mle_analysis_dict = self.aggregate_optima(models=self.models[i], sig=sig, decimals=decimals)
                self.mle_analysis.append([mle_model, mle_analysis_dict])
                
                # Compares BICs
                self.metrics.append(mle_model.model_metrics[self.comparison_metric])

                # Usually minimising, i.e. BIC, AIC, Neg log likelihood
                if mle_model.model_metrics[self.comparison_metric] < self.mle_model.model_metrics[self.comparison_metric]:
                        self.mle_model = mle_model                

            print(f'\nMLE has {self.mle_model.c} clusters. \n')

        # If summary is true, print a summary of the MLE model
        if summary:
                print(f'\n Using {self.comparison_metric} for model selection... \n')
                self.mle_model.summary(sig=0) 

        # If assignment is true, return a vector with the assignement
        if assignment:
            return self.mle_model.assignments


    


        

        

    



        


                

