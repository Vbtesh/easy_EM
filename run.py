import numpy as np
import pandas as pd
import warnings
from methods.debug import handle_warning
# Handle runtime warning
#warnings.showwarning = handle_warning
warnings.simplefilter('error')
from classes.model import Model
from classes.collection import Collection

# Input file path of data set or python object
# If csv, 
file_path = '.\\data\\intervention_df.csv'
file_path = '.\\data\\participant_df.csv'
#file_path = '.\\data\\prior_df.csv'

if file_path[-3:] == 'csv':
    dataset = pd.read_csv(file_path)
    df = True

# Var dict shape
# Dict of dict
# Keys are variable names (same as in the dataset) or column index (if dataset is np.array)
## Secondary keys and values: 
### var_type: poisson, multinomial, gaussian_mean, etc
### params: None or vector/matrix of parameters appropriate for the distribution
### add_params: Additional parameters when applicable: num outcomes for multinomial, variance for gaussian_mean

var_dict_int = {
    'range': {
        'type': 'gaussian_mean',
        'params': None,
        'add_params': None
    },
    'length': {
        'type': 'gaussian_mean',
        'params': None,
        'add_params': None
    }
}

var_dict_part = {
    'int_avg_range': {
        'type': 'gaussian_mean',
        'params': None,
        'add_params': None
    },
    'int_avg_length_sec': {
        'type': 'gaussian_mean',
        'params': None,
        'add_params': None
    },
    'inters_real': {
        'type': 'gaussian_mean',
        'params': None,
        'add_params': None
    },
    'meanErr': {
        'type': 'gaussian_mean',
        'params': None,
        'add_params': None
    }
}

keys = dataset.drop(columns=['cond', 'dataValid']).columns.to_list()[1:]
value = {
    'type': 'multinomial',
    'params': None,
    'add_params': 5
}

var_dict_prior = {col:value for col in keys}

num_clusters = [6, 7, 8, 9, 10, 11]
n_iter = 200
sig = 0.99

N = 50

coll = Collection('3_c_4_c', N, var_dict_part, dataset, num_clusters, hidden_init='random')

coll.train_models(n_iter)
 
assigns = coll.maximum_likelihood_estimation(sig=sig, comparison_metric='aicc')

pass