# this is the the main file for the spectral project

import os
import pandas as pd
import csv

from plotting_functions import plot_row_by_row
from basic_fitting_attempts import fit_regression_raw_by_raw
from basic_fitting_attempts import model_training_wrapper
from basic_fitting_attempts import model_training_wrapper_2
from preprocessing_module import initial_formatting_1
from preprocessing_module import convert_for_n_order_modelling
from preprocessing_module import splitting_wrapper
from preprocessing_module import down_sample
from dnn_regression import dnn_regression_wrapper
from dnn_regression import lstm_wrapper

path = '/Users/sven/kohalikTree/Data/AIRSCS/spectral/'
#model_order = 25   # 25 low res
model_order = 55
#load two files
initial_data_file = path + 'sarspec_X.csv'
target_data_file = path + 'lainespec_Y.csv'

initial_data = pd.read_csv(initial_data_file, sep=';')
target_data = pd.read_csv(target_data_file, sep=';')

#fit_regression_raw_by_raw(initial_data, target_data)

initial_data, target_data = initial_formatting_1(initial_data, target_data)
initial_data = down_sample(initial_data, target_data)
#dnn_regression_wrapper(initial_data, target_data, model_order)
lstm_wrapper(initial_data, target_data)

#model_training_wrapper_2(initial_data, target_data, model_order)
#model_training_wrapper(initial_data, target_data, model_order)

#initial_data_train, initial_data_test, target_data_train, target_data_test = splitting_wrapper(initial_data, target_data)

#dict_of_observations, u_combined, y_combined = convert_for_n_order_modelling(initial_data, target_data, model_order)


#plot_row_by_row(initial_data, target_data)
#print("That's all folks!!!")

