# this file is to perform pseudo tokenization for the numeric data it consists of two functions
# one is actual tokenization and the the other is to determine the range of the files

import numpy as np


def return_range(vocab_size, *args):
    # each arg is pandas dataframe where row corresponds to the obs point to comput max and min
    min_val = []
    max_val = []
    for arg in args:
        min_val.append(np.min(arg))
        max_val.append(np.max(arg))
    bins = np.linspace(np.min(min_val), np.max(max_val), vocab_size)
    return bins


def tokenize_numeric(data, bins):
    rows, cols = data.shape
    data_1 = np.zeros((rows, cols)).astype(int)
    for i in range(0, rows):
        data_1[i, :] = np.digitize(data[i, :], bins, right=False)
        #data_1 = data.astype(int)

    return data_1


def token2numeric(data, bins):
    rows, cols = data.shape
    data_1 = np.zeros((rows, cols))
    for i in range(0, rows):
        for j in range(0, cols):
            data_1[i, j] = bins[data[i, j]-1]

    return data_1