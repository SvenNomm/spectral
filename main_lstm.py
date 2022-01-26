# this is the the main file for  lstm case the spectral project
# NB case of the intial dataset.
# for the new data function initial data formatting should be changed

import pandas as pd

#from preprocessing_module import initial_formatting_1  # Reading and formating of the initial data
from preprocessing_module import initial_formatting_2  # use this line for the second generation of the  data set
from preprocessing_module import initial_formatting_3  # use this line for the third generation of the  data set
from preprocessing_module import down_sample # this function equalizes the lengthes of the sequences
from dnn_regression import lstm_wrapper
from preprocessing_module import delete_nan_rows

#path = '/Users/sven/kohalikTree/Data/AIRSCS/spectral/'
path = '/Users/sven/kohalikTree/Data/AIRSCS/spectral_1/'  # to read data submitted on 10.12.2021
path = '/Users/sven/kohalikTree/Data/AIRSCS/spectral_2/'  # this is for the large data set provided end 2021-jan2022

#model_order = 55
#load the files
#initial_data_file = path + 'sarspec_X.csv'
#target_data_file = path + 'lainespec_Y.csv'

#initial_data_file = path + 'sarspec_hgh_clean.csv' # this and the following row are for the data 10.12.2021
#target_data_file = path + 'wavespec_hgh_clean.csv'

initial_data_file = path + 'X30_sarspec_hgh_order_2_winx_1024_clean.csv' # this and the following row are for the data xx.12.2021 - xx.01.2022
target_data_file = path + 'X30_wavespec_hgh_order_2_winx_1024_clean.csv'


initial_data = pd.read_csv(initial_data_file, sep=',') # keep in mind which separator to use
target_data = pd.read_csv(target_data_file, sep=',')

initial_data, target_data = delete_nan_rows(initial_data, target_data)

initial_data, target_data = initial_formatting_3(initial_data, target_data)
#initial_data = down_sample(initial_data, target_data)
lstm_wrapper(initial_data, target_data)

print("That's all folks!!!")

