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

    print("Initial  formatting has been completed!")
    return initial_data, target_data


def initial_formatting_2(initial_data, target_data):
    print("Aloha! Performing initial formatting")

    # the following two lines are very local
    del initial_data['time']
    del target_data['time']
    del target_data['HS']

    #target_data.drop(target_data.index[0], inplace=True)
    #target_data = target_data.reset_index()
    #del target_data['index']

    print("Initial  formatting has been completed!")
    return initial_data, target_data

def initial_formatting_3(initial_data, target_data):
    print("Aloha! Performing initial formatting")

    # the following lines are very local
    del initial_data['time']
    del target_data['time']
    del target_data['hs']
    del target_data['tp']
    del target_data['tc']
    del target_data['tm_10']
    del target_data['tm01']
    del target_data['tm02']
    del target_data['kappa']
    del target_data['meandir']
    del target_data['pdir']
    del target_data['meanspread']
    del target_data['pspread']

    #target_data.drop(target_data.index[0], inplace=True)
    #target_data = target_data.reset_index()
    #del target_data['index']

    print("Initial  formatting has been completed!")
    return initial_data, target_data


def delete_nan_rows(df1, df2):
    selected_rows = df1.loc[df1.isna().any(axis=1)].index.tolist()
    df1 = df1.drop(df1.index[selected_rows])
    df2 = df2.drop(df2.index[selected_rows])
    return df1, df2

def splitting_wrapper(initial_data, target_data):
    print("Hello, this is splitting wrapper!")
    _, cols = initial_data.shape

    target_columns = target_data.columns
    initial_columns = initial_data.columns
    merged_df = pd.concat([initial_data, target_data], axis=1)
    train_df, test_df = train_test_split(merged_df, test_size=0.3)
    initial_data_train = train_df.iloc[:, 0: cols]
    initial_data_test = test_df.iloc[:, 0: cols]
    target_data_train = train_df.iloc[:, cols: cols * 2]
    target_data_test = test_df.iloc[:, cols: cols * 2]

    initial_data_train = initial_data_train.reset_index()
    del initial_data_train['index']

    initial_data_test = initial_data_test.reset_index()
    test_index = initial_data_test['index']
    del initial_data_test['index']

    target_data_train = target_data_train.reset_index()
    del target_data_train['index']

    target_data_test = target_data_test.reset_index()
    del target_data_test['index']
    print("Splitting has been completed.")
    return initial_data_train, initial_data_test, target_data_train, target_data_test, test_index


def down_sample(initial_data, target_data):
    print("Performing down sapling!")
    columns = target_data.columns
    rows_i, cols_i = initial_data.shape
    rows_t, cols_t = target_data.shape
    x = np.empty((rows_i, cols_t))
    if cols_i > cols_t:
        for i in range(0, rows_i):
            xx = initial_data.loc[i, :].to_numpy()
            x[i, :] = np.abs(resample(xx, num=cols_t).reshape(1, cols_t))

        x = pd.DataFrame(x, columns=columns)
    print("Down sampling has sbeen completed.")
    return x

def up_sample(initial_data, target_data):
    columns = initial_data.columns
    rows_i, cols_i = initial_data.shape
    rows_t, cols_t = target_data.shape
    x = np.empty((rows_i, cols_i))
    if cols_i < cols_t:
        for i in range(0, rows_i):
            xx = initial_data.loc[i - 1, :].to_numpy()
            x[i, :] = resample(x, num=len(cols_t))

    initial_data = pd.DataFrame(x, columns=columns)
    return initial_data


def apply_log(data):
    print("transfering to the log scale!")
    rows = len(data)
    for column in data.columns:
        for i in range(0, rows):
            data.loc[i, column] = np.log(data.loc[i, column])
    print("log scaling has been completed.")
    return data


def apply_normalization(data):
    print("Normalizing the data!")
    rows, cols = data.shape
    for j in range(0, cols):
        col_min = np.min(data[:, j])
        col_max = np.max(data[:, j])
        col_ampl = col_max - col_min
        print(col_ampl)
        for i in range(0, rows):
            data[i,j] = ( data[i,j] - col_min) / col_ampl
    print("Data has been normalized!")
    return data


def convert_for_n_order_modelling(initial_data, target_data, model_order):
    print("Converting for n-order modelling!")
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
    print("Ready for n-order modelling")
    return dict_of_observations, u_combined, y_combined


def convert_for_n_order_modelling_upsampling(initial_data, target_data, model_order):
    print("Converting for n- order modelling with up-sumpling")
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
    print("Ready for n-order modelling with up sumpling!")
    return dict_of_observations, u_combined, y_combined


def alternative_formatting_for_modelling(dict_of_observations, model_order):
    print("Alternative formatting")
    dict_keys = list(dict_of_observations.keys())
    #print(dict_keys[0])
    rows = len(dict_of_observations[dict_keys[0]])
    data_set = np.empty((0, model_order+1))

    for i in range(0, rows):
        for key in dict_keys:
            a = dict_of_observations[key][i, :]
            data_set = np.append(data_set, dict_of_observations[key][i, :].reshape(1,model_order+1), axis=0)
    print("Done!")
    return data_set


def data_formatter(initial_data, target_data, model_order):
    print("Aloha! I am going to format the data for dnn!!!")
    # Make NumPy printouts easier to read.

    #initial_data = apply_log(initial_data)
    #target_data = apply_log(target_data)

    initial_data_train, initial_data_test, target_data_train, target_data_test, test_index = splitting_wrapper(initial_data,
                                                                                                   target_data)

    dict_train, _, _ = convert_for_n_order_modelling(initial_data_train, target_data_train, model_order)
    dict_test, u_test, y_test = convert_for_n_order_modelling(initial_data_test, target_data_test, model_order)

    # dict_train, _, _ = convert_for_n_order_modelling_upsampling(initial_data_train, target_data_train, model_order)
    # dict_test, u_test, y_test = convert_for_n_order_modelling_upsampling(initial_data_test, target_data_test, model_order)

    data_train = alternative_formatting_for_modelling(dict_train, model_order)
    data_test = alternative_formatting_for_modelling(dict_test, model_order)
    print("Data is ready for dnn!")
    return data_train, data_test, dict_train, dict_test, test_index
