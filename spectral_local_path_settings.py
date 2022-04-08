# actually this file should not be shared over git as it contains local path information

def return_path():
    initial_data_path = '/home/sven/kohalikTree/Data/AIRSCS/spectral_1/'
    preprocessed_data_path = '/home/sven/kohalikTree/Data/AIRSCS/spectral_1/'
    return initial_data_path, preprocessed_data_path


def return_model_path():
    model_path = '/home/sven/kohalikTree/Data/AIRSCS/spectral_1/'
    return model_path

def return_file_names():
    #initial_data_file = 'sarspec_hgh_clean.csv'  # this and the following row are for the data 10.12.2021
    #target_data_file = 'wavespec_hgh_clean.csv'

    #initial_data_file = 'X30_sarspec_hgh_order_2_winx_1024_clean.csv'  # this and the following row are for the data 10.12.2021
    #target_data_file = 'X30_wavespec_hgh_order_2_winx_1024_clean.csv'

    return initial_data_file, target_data_file


def return_processed_file_names(pp_type):
    initial_data_file, target_data_file = return_file_names()
    initial_name_base = initial_data_file.split('.')[0]  # extract name for the reference
    target_name_base = target_data_file.split('.')[0]

    path, processed_data_path = return_path()

    initial_data_train_fname = processed_data_path + initial_name_base + pp_type + 'initial_train.pkl'
    target_data_train_fname = processed_data_path + target_name_base + pp_type + 'target_train.pkl'
    initial_data_valid_fname = processed_data_path + initial_name_base + pp_type + 'initial_valid.pkl'
    target_data_valid_fname = processed_data_path + target_name_base + pp_type + 'target_valid.pkl'
    valid_data_index_fname = processed_data_path + target_name_base + pp_type + 'target_index.pkl'
    print("I am about to load the following files:")
    print(initial_data_train_fname)
    print(target_data_train_fname)
    print(initial_data_valid_fname)
    print(target_data_valid_fname)
    print(valid_data_index_fname)
    return initial_data_train_fname, target_data_train_fname, initial_data_valid_fname, target_data_valid_fname, valid_data_index_fname

