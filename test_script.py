from methods.tests import generate_test_data, cluster_agreement
import numpy as np
import pandas as pd
from classes.model import Model

n = 60
c = 4
variables = [
    {
        'name': 'brain_to_body',
        'type': 'gaussian_mean',
        'params': 30,
        'clustered': 0.3
    },
    {
        'name': 'num_successes',
        'type': 'poisson',
        'params': 100,
        'clustered': 0.3
    },
    {
        'name': 'colour',
        'type': 'multinomial',
        'params': 3,
        'clustered': True
    }
]

assignment, params, dataset = generate_test_data(n, c, variables, df=False)
 
var_dict = {
    'brain_to_body': {
        'type': 'gaussian_mean',
        'params': None,
        'add_params': None
    },
    'num_successes': {
        'type': 'poisson',
        'params': None,
    },
    'colour': {
        'type': 'multinomial',
        'params': None,
        'add_params': 3
    }
}
var_dict = {
    0: {
        'type': 'gaussian_mean',
        'params': None,
        'add_params': None
    },
    1: {
        'type': 'poisson',
        'params': None,
    },
    2: {
        'type': 'multinomial',
        'params': None,
        'add_params': 3
    }
}


model = Model('test', c, var_dict, data=dataset, df=False)

model.run_EM(n_iter = 400)

q_h = model.hidden.q_h
sig = 0.75

prob_assign = np.argmax(q_h, axis=1)
prob_max = np.amax(q_h, axis=1)
tsh = np.where(prob_max < sig)[0]
prob_assign[tsh] = c+1

cluster_success, inv_cluster_success, unclassified_rate, assign_clusters, learn_clusters, unclassified = cluster_agreement(assignment, prob_assign, c)

for i, j in enumerate(cluster_success):
    print('Original cluster:', i, 'Classified cluster:', j[0], 'Correctness rate:', j[1])

print('Number of unclassified:', len(unclassified), 'Unclassification rate:', unclassified_rate)

model.summary(sig=sig)
pass