# actually this file should not be shared over git as it contains local path information

def return_path():
    initial_data_path = '/Users/svennomm/kohalikTree/Data/AIRSCS/spectral_1/'
    preprocessed_data_path = '/Users/svennomm/kohalikTree/Data/AIRSCS/spectral_1/'
    return initial_data_path, preprocessed_data_path

def return_file_names():
    initial_data_file = 'sarspec_hgh_clean.csv'  # this and the following row are for the data 10.12.2021
    target_data_file = 'wavespec_hgh_clean.csv'

    return initial_data_file, target_data_file
