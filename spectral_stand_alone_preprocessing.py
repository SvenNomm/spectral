# this code preprocessess the data of the spectral project

import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
from spectral_local_path_settings import *
import matplotlib.pyplot as plt
import numpy as np
import torch


def apply_normalization(data):
    print("Normalizing the data!")
    rows, cols = data.shape
    for j in range(0, cols):
        col_min = np.min(data.iloc[:, j])
        col_max = np.max(data.iloc[:, j])
        col_ampl = col_max - col_min
        #print(col_ampl)
        for i in range(0, rows):
            data.iloc[i,j] = (data.iloc[i,j] - col_min) / col_ampl
    print("Data has been normalized!")
    return data


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

#def apply_log(data):
#    print("transfering to the log scale!")
#    columns = data.columns

#    data_1 = tf.convert_to_tensor(data)
#    data_1 = tf.math.log(data_1)

#    data_1 = data_1.numpy()
#    data_1 = pd.DataFrame(data_1, columns=columns)
#    #rows = len(data)
#    #for column in data.columns:
    #    for i in range(0, rows):
    #        data.loc[i, column] = np.log(data.loc[i, column])
#    print("log scaling has been completed.")
#    return data_1


def apply_log(data):
    print("Transfering to log scale.")
    columns = data.columns
    data_1 = torch.tensor(data.values)
    data_1 = torch.log(data_1)
    data = pd.DataFrame(data_1, columns=columns)
    return data






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

#pp_type = '_raw_'
#pp_type = '_normalized_'
pp_type = '_log_scale_'
#pp_type = '_norm_log_'
#pp_type = '_log_norm_'

path, processed_data_path = return_path()
initial_data_file, target_data_file = return_file_names() # get the file names

initial_name_base = initial_data_file.split('.')[0] # extract name for the reference
target_name_base = target_data_file.split('.')[0]

initial_data_file = path + initial_data_file # this and the following row are for the data 10.12.2021
target_data_file = path + target_data_file
initial_data = pd.read_csv(initial_data_file, sep=',') # keep in mind which separator to use
target_data = pd.read_csv(target_data_file, sep=',')

#fig = plt.figure()
#for i in range(0, len(initial_data)):
#    plt.plot(initial_data.loc[i, :], color='blue', linewidth=0.1)
#    plt.plot(target_data.loc[i, :], color='yellow', linewidth=0.1)

#plt.show()

initial_data, target_data = initial_formatting_3(initial_data, target_data)
initial_data = apply_log(initial_data)
target_data = apply_log(target_data)
#initial_data = apply_normalization(initial_data)
#target_data = apply_normalization(target_data)


#fig = plt.figure()
#for i in range(0, len(initial_data)):
#    plt.plot(initial_data.loc[i, :], color='blue', linewidth=0.1)
#    plt.plot(target_data.loc[i, :], color='yellow', linewidth=0.1)

#plt.show()

initial_data_train, initial_data_test, target_data_train, target_data_test, test_index = splitting_wrapper(initial_data, target_data)


initial_data_train = initial_data_train.to_numpy()
output = open(processed_data_path + initial_name_base + pp_type +'initial_train.pkl', 'wb')
pickle.dump(initial_data_train, output)
output.close()
target_data_train = target_data_train.to_numpy()
output = open(processed_data_path + target_name_base + pp_type + 'target_train.pkl', 'wb')
pickle.dump(target_data_train, output)
output.close()
initial_data_test = initial_data_test.to_numpy()
output = open(processed_data_path + initial_name_base + pp_type + 'initial_valid.pkl', 'wb')
pickle.dump(initial_data_test, output)
output.close()
target_data_test = target_data_test.to_numpy()
output = open(processed_data_path + target_name_base + pp_type + 'target_valid.pkl', 'wb')
pickle.dump(target_data_test, output)
output.close()
output = open(processed_data_path + target_name_base + pp_type + 'target_index.pkl', 'wb')
pickle.dump(test_index, output)
output.close()