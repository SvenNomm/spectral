# this file contains plotting functions

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def plot_row_by_row(initial_data, target_data):
    rows = len(target_data)

    # the following two lines are very local
    del initial_data['k']
    del target_data['f']

    target_columns = target_data.columns
    initial_columns = initial_data.columns

    initial_columns = initial_columns[0 : len(target_columns)]

    fig1, axis = plt.subplots()

    for i in range(1, rows):
        b = initial_data.loc[i-1, initial_columns].to_numpy()
        plt.plot(initial_data.loc[i-1, initial_columns].to_numpy(), color='green')
        plt.plot(target_data.loc[i, :].to_numpy(), color='orange')
        plt.show()



