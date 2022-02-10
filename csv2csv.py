# this is stand alone code to load and save csvs for the new version of pandas

import pandas as pd


#path = '/Users/sven/kohalikTree/Data/AIRSCS/spectral/'
path = '/Users/sven/kohalikTree/Data/AIRSCS/spectral_1/'  # to read data submitted on 10.12.2021
path = '/Users/sven/kohalikTree/Data/AIRSCS/spectral_2/'  # this is for the large data set provided end 2021-jan2022
path_2 = '/Users/sven/kohalikTree/Data/AIRSCS/spectral_2_reprocessed/'
file_name_initial = 'X30_sarspec_hgh_order_2_winx_1024_clean'
file_name_target = 'X30_wavespec_hgh_order_2_winx_1024_clean'

initial_data_file = path + file_name_initial + '.csv'
target_data_file = path + file_name_target + '.csv'

initial_data = pd.read_csv(initial_data_file, sep=',') # keep in mind which separator to use
target_data = pd.read_csv(target_data_file, sep=',')

initial_data_file = path + file_name_initial + '_reprocessed_' + '.csv' # this and the following row are for the data xx.12.2021 - xx.01.2022
target_data_file = path + file_name_target + '_reprocessed_' + '.csv'

initial_data.to_csv(initial_data_file, encoding='utf-8', index=False, sep=',')
target_data.to_csv(target_data_file, encoding='utf-8', index=False, sep=',')

del initial_data, target_data
initial_data = pd.read_csv(initial_data_file, sep=',') # keep in mind which separator to use
target_data = pd.read_csv(target_data_file, sep=',')

print("That's all filks!!!")