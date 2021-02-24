import numpy as np
import pandas as pd
from methods.general import normArray
# Generate a test dataset


def generate_test_data(n, clusters, variables, df=False):

    assignment = np.random.randint(clusters, size=n)

    # variables list of dictionary structure
    ## indices: var_indices
    ## values: dict
    ### keys, values:
    #### name, name of the variable
    #### type, type of distribution (e.g. height_gaussian_mean)
    #### params, mean parameters (mean for gaussian, rate for poisson, outcomes for multinomial)
    #### clustered, ratio or False, if ratio, generate as many parameters set as cluster and uses ratio as a measure of deviation from the mean


    # Choose relevant variables, maybe some variables are the same across the sample
    # This just implies the number of clusters within each variable, we start with the easy case 
    # Define a noise factor (how noisy the clusters are)
    # How far apart the distributions of each clusters are

    # For each variable, given the relevant number of clusters, generate that number of parameters.
    # These parameters should be drawn from a normal distribution with given mean and variance noise factor

    params = [None for _ in range(len(variables))]
    
    dataset = np.zeros((n, len(variables)))

    for k, infos in enumerate(variables):

        name = infos['name']
        var_type = infos['type']
        spread_ratio = infos['clustered']

        if var_type == 'gaussian_mean':
            if spread_ratio:
                spread = np.abs(infos['params'] * spread_ratio)
                params[k] = np.random.normal(loc=infos['params'], scale=spread, size=clusters) 

                # Fill dataset
                dataset[: , k] = np.random.normal(loc=params[k][assignment], scale=1)

            else:
                params[k] = infos['params']

                # Fill dataset
                dataset[: , k] = np.random.normal(loc=params[k], scale=1, size=n)


        elif var_type == 'poisson':
            if spread_ratio:
                spread = np.abs(infos['params'] * spread_ratio)
                params[k] = np.random.normal(loc=infos['params'], scale=spread, size=clusters) 

                # Fill dataset
                dataset[: , k] = np.random.poisson(lam=params[k][assignment])
            else:
                params[k] = infos['params']

                # Fill dataset
                dataset[: , k] = np.random.poisson(lam=params[k], size=n)

            
            
        elif var_type == 'multinomial':
            if spread_ratio:
                params[k] = np.random.uniform(size=(clusters, infos['params']))
                params[k] = normArray(params[k])

                # Fill dataset
                for i in range(n):
                    dataset[i , k] = np.random.choice(infos['params'], size=1, p=params[k][assignment[i]])
            else:
                params[k] = np.random.uniform(size=infos['params'])
                params[k] = params[k] / params[k].sum()

                # Fill dataset
                for i in range(n):
                    dataset[i , k] = np.random.choice(infos['params'], size=1, p=params[k])

    # Return the appropriate data structure
    if not df:
        return (assignment, params, dataset)
    else:
        cols = [k['name'] for k in variables]
        df = pd.DataFrame()
        
        for k, infos in enumerate(variables):
            df[infos['name']] = dataset[:, k]

        return (assignment, params, df)


def cluster_agreement(assign, learn, num_clusters):
    assign_clusters = []
    assign_clusters_len = []
    for i in range(num_clusters):
        assign_clusters.append(np.where(assign == i)[0])
        assign_clusters_len.append(len(assign_clusters[i]))

    learn_clusters = []
    learn_clusters_len = []
    for i in range(num_clusters):
        learn_clusters.append(np.where(learn == i)[0])
        learn_clusters_len.append(len(learn_clusters[i]))

    unclassified = np.where(learn == num_clusters + 1)[0]
    unclassified_rate = len(unclassified) / len(learn)

    # Compare clusters
    cluster_success = []
    
    for i in range(num_clusters):
        best = 0
        best_idx = 0
        for j in range(num_clusters):
            agree = np.intersect1d(assign_clusters[i], learn_clusters[j])
            if len(agree) / len(assign_clusters[i]) > best:
                best = len(agree) / len(assign_clusters[i]) 
                best_idx = j

        cluster_success.append([best_idx, best])

    # Compare clusters
    inv_cluster_success = []
    
    for i in range(num_clusters):
        best = 0
        best_idx = 0
        for j in range(num_clusters):
            agree = np.intersect1d(assign_clusters[j], learn_clusters[i])
            if len(learn_clusters[i]) > 0:
                if len(agree) / len(learn_clusters[i]) > best:
                    best = len(agree) / len(learn_clusters[i]) 
                    best_idx = j
                

        inv_cluster_success.append([best_idx, best])

    return [cluster_success, inv_cluster_success, unclassified_rate, assign_clusters, learn_clusters, unclassified]


        


    



