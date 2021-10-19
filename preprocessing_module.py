# this file contains the functions to preprocess the data

import pandas as pd
import numpy as np
from scipy.signal import resample
from sklearn.model_selection import train_test_split


def initial_formatting_1(initial_data, target_data):
    print("Aloha! Performing initial formatting")

    # the following two lines are very local
    del initial_data['k']
    del target_data['f']

    target_data.drop(target_data.index[0], inplace=True)
    target_data = target_data.reset_index()
    del target_data['index']

    return initial_data, target_data


def splitting_wrapper(initial_data, target_data):
    target_columns = target_data.columns
    initial_columns = initial_data.columns
    merged_df = pd.concat([initial_data, target_data], axis=1)
    train_df, test_df = train_test_split(merged_df, test_size=0.3)
    initial_data_train = train_df[initial_columns]
    initial_data_test = test_df[initial_columns]
    target_data_train = train_df[target_columns]
    target_data_test = test_df[target_columns]

    initial_data_train = initial_data_train.reset_index()
    del initial_data_train['index']

    initial_data_test = initial_data_test.reset_index()
    del initial_data_test['index']

    target_data_train = target_data_train.reset_index()
    del target_data_train['index']

    target_data_test = target_data_test.reset_index()
    del target_data_test['index']

    return initial_data_train, initial_data_test, target_data_train, target_data_test


def apply_log(data):
    rows = len(data)
    for column in data.columns:
        for i in range(0, rows):
            data.loc[i, column] = np.log(data.loc[i, column])
    return data

def convert_for_n_order_modelling(initial_data, target_data, model_order):
    target_columns = target_data.columns
    initial_columns = initial_data.columns
    rows = len(target_data)
    cols = len(target_columns)
    dict_of_observations = {}
    u_combined = np.empty((0, model_order-1))
    y_combined = np.empty((0))

    for i in range(1, rows-1):
        #print("------------------------------------------------")
        x = initial_data.loc[i - 1, :].to_numpy()
        print(np.argwhere(np.isnan(x)))

        l = len(target_columns)
        x = resample(x, num=len(target_columns)) # x is resampled because of its length
        print("i =", i, "NaN indexes in x are", np.argwhere(np.isnan(x)))

        y = target_data.loc[i, :].to_numpy()
        print("Analyzing y")
        print(np.argwhere(np.isnan(y)))

        u = np.zeros((cols - model_order, model_order + 1))
        for j in range(0, model_order):
            u[:, j] = x[j: cols - model_order + j]
            #print(j, "u", model_order, "///",
            #      i)

            #print(np.argwhere(np.isnan(u)))

            #print(">>>>>>>>>>>>>>>")

        u[:, model_order] = y[model_order: cols]

        u_combined = np.append(u_combined, u[:, 0:model_order-1], axis=0)
        y_combined = np.append(y_combined, y[model_order: cols], axis=0)

        dict_of_observations[i] = u
    #print("merged ------")
    #print(np.argwhere(np.isnan(u_combined)))
    #print(np.argwhere(np.isnan(y_combined)))
    print("That's all folks!!!")
    return dict_of_observations, u_combined, y_combined


def convert_for_n_order_modelling_upsampling(initial_data, target_data, model_order):
    target_columns = target_data.columns
    initial_columns = initial_data.columns
    rows = len(target_data)
    cols = len(initial_columns)
    dict_of_observations = {}
    u_combined = np.empty((0, model_order-1))
    y_combined = np.empty((0))

    for i in range(1, rows-1):
        #print("------------------------------------------------")
        x = initial_data.loc[i - 1, :].to_numpy()

        y = target_data.loc[i, :].to_numpy()
        y = resample(y, num=cols)
        u = np.zeros((cols - model_order, model_order + 1))
        for j in range(0, model_order):
            u[:, j] = x[j: cols - model_order + j]
            #print(j, "u", model_order, "///",
            #      i)

            #print(np.argwhere(np.isnan(u)))

            #print(">>>>>>>>>>>>>>>")

        u[:, model_order] = y[model_order: cols]

        u_combined = np.append(u_combined, u[:, 0:model_order-1], axis=0)
        y_combined = np.append(y_combined, y[model_order: cols], axis=0)

        dict_of_observations[i] = u
    #print("merged ------")
    #print(np.argwhere(np.isnan(u_combined)))
    #print(np.argwhere(np.isnan(y_combined)))
    print("That's all folks!!!")
    return dict_of_observations, u_combined, y_combined


def alternative_formatting_for_modelling(dict_of_observations, model_order):
    dict_keys = list(dict_of_observations.keys())
    #print(dict_keys[0])
    rows = len(dict_of_observations[dict_keys[0]])
    data_set = np.empty((0, model_order+1))

    for i in range(0, rows):
        for key in dict_keys:
            a = dict_of_observations[key][i, :]
            data_set = np.append(data_set, dict_of_observations[key][i, :].reshape(1,model_order+1), axis=0)

    return data_set
