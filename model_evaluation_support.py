# this file contains the functions for model quality evaluation.

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error


def goodness_descriptor(test_y, hat_y):
    mse = mean_squared_error(test_y, hat_y)
    rho = np.corrcoef(test_y, hat_y)[0, 1]
    max_test = test_y.argmax(axis=0)
    max_hat = hat_y.argmax(axis=0)
    delta_max_val = np.max(test_y) - np.max(hat_y)
    delta_max_loc = max_test - max_hat

    return mse, rho, max_test, max_hat, delta_max_val, delta_max_loc


