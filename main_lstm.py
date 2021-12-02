# this is the the main file for  lstm case the spectral project
# NB case of the intial dataset.
# for the new data function initial data formatting should be changed

import pandas as pd

from preprocessing_module import initial_formatting_1  # Reading and formating of the initial data
from preprocessing_module import down_sample # this function equalizes the lengthes of the sequences
from dnn_regression import lstm_wrapper

path = '/Users/sven/kohalikTree/Data/AIRSCS/spectral/'
model_order = 55
#load the files
initial_data_file = path + 'sarspec_X.csv'
target_data_file = path + 'lainespec_Y.csv'

initial_data = pd.read_csv(initial_data_file, sep=';')
target_data = pd.read_csv(target_data_file, sep=';')

initial_data, target_data = initial_formatting_1(initial_data, target_data)
initial_data = down_sample(initial_data, target_data)
lstm_wrapper(initial_data, target_data)

print("That's all folks!!!")
