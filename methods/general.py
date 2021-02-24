import numpy as np

# Function that normalise an A along the desired axis, default is rows
def normArray(A, axis=1):
    norm_A = A.sum(axis=axis)

    if axis == 1:
        return A / norm_A[:, np.newaxis]
    else:
        return A / norm_A

# Simple function that returns i random discrete probability distribution over j categories
def genDiscreteDist(i=1, j=3):
    d = np.random.uniform(size=(i, j))
    norm_d = d.sum(axis=1)

    dNorm = d / norm_d[:, np.newaxis]

    return dNorm