# this file contains functions to perform deep neural networks based regression

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from preprocessing_module import apply_log
from preprocessing_module import splitting_wrapper
from preprocessing_module import convert_for_n_order_modelling
from preprocessing_module import convert_for_n_order_modelling_upsampling
from preprocessing_module import alternative_formatting_for_modelling
from preprocessing_module import data_formatter
from dnn_support import train_model
from dnn_support import test_model


def build_and_compile_model(norm):
    print("Building convolutional model!")
    model = keras.Sequential([
        norm,
        layers.Dense(55, activation='sigmoid'),   # 55, 25, 1 works well with model order 5
        layers.Dense(25, activation='sigmoid'),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(loss='mean_absolute_error',
                  optimizer=tf.keras.optimizers.Adam(0.001))
    print("Convolutional model is ready!")
    return model


def dnn_regression_wrapper(initial_data, target_data, model_order):
    print("Aloha! this is dnn_regression_wrapper!!!")
    # Make NumPy printouts easier to read.
    test_results = {}

    np.set_printoptions(precision=3, suppress=True)
    print(tf.__version__)
    data_train, data_test, dict_train, dict_test = data_formatter(initial_data,target_data, model_order)

    train_features = data_train[:, 0:model_order - 1]
    train_labels = data_train[:, model_order]

    test_features = data_test[:, 0:model_order - 1]
    test_labels = data_test[:, model_order]

    normalizer = tf.keras.layers.Normalization(axis=-1)
    normalizer.adapt(train_features)

    dnn_model = build_and_compile_model(normalizer)
    dnn_model.summary()
    history = dnn_model.fit(
        train_features,
        train_labels,
        validation_split=0.2,
        verbose=0, epochs=100)

    test_results['dnn_model'] = dnn_model.evaluate(test_features, test_labels, verbose=0)
    TR = pd.DataFrame(test_results, index=['Mean absolute error']).T

    test_keys = dict_test.keys()
    for key in test_keys:
        data_test = dict_test[key]
        test_features = data_test[:, model_order-1]
        test_labels = data_test[:, model_order]
        y_ampl = np.abs(np.max(test_labels) - np.min(test_labels))
        test_predictions = dnn_model.predict(test_features).flatten()

        residuals_nn = (test_labels - test_predictions) / y_ampl

        fig2, axis = plt.subplots()
        plt.plot(test_labels, color='blue')
        plt.plot(test_predictions, color='orange')
        plt.title("validation on a small set")
        plt.show()

        fig3, axis = plt.subplots()
        plt.plot(residuals_nn, color='green')
        plt.title("residuals for a small set")
        plt.show()
        print("It was convolutional wrapper!")


def lstm_wrapper(initial_data, target_data):
    print("Aloha! this is lstm_wrapper!!!")

    np.set_printoptions(precision=3, suppress=True)
    print(tf.__version__)
    initial_data_train, initial_data_valid, target_data_train, target_data_valid, valid_index = splitting_wrapper(initial_data, target_data)
    initial_data_train, initial_data_test, target_data_train, target_data_test, no_need_index = splitting_wrapper(initial_data_train, target_data_train)
    initial_data_test.to_csv('test_x_08_04_2022_1.csv', sep='\t', encoding='utf-8')
    target_data_test.to_csv('test_y_08_04_2022_1.csv', sep='\t', encoding='utf-8')
    valid_index.to_csv('test_index_08_04_2022_1.csv', sep='\t', encoding='utf-8')

    #data_train, data_test, dict_train, dict_test = data_formatter(initial_data, target_data, model_order)

    #train_features = data_train[:, 0:model_order - 1] #inputs
    #train_labels = data_train[:, model_order] #outputs

    #train_features = data_train[:, 0:model_order]  # inputs
    #train_labels = data_train[:, model_order]  # outputs

    lstm_model = train_model(initial_data_train, target_data_train)
    lstm_model.save('lstm_model_28_01_2022_1.h5')

    test_model(initial_data_valid, target_data_valid, lstm_model, valid_index)
    print("Uhhh managed the task")



    