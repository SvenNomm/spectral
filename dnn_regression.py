# this file contains functions to perform deep neural networks based regression

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from preprocessing_module import apply_log
from preprocessing_module import splitting_wrapper
from preprocessing_module import convert_for_n_order_modelling
from preprocessing_module import convert_for_n_order_modelling_upsampling
from preprocessing_module import alternative_formatting_for_modelling


def build_and_compile_model(norm):
    model = keras.Sequential([
        norm,
        layers.Dense(5, activation='sigmoid'),   # 55, 25, 1 works well with model order 5
        layers.Dense(3, activation='relu'),
        layers.Dense(1)
    ])

    model.compile(loss='mean_absolute_error',
                  optimizer=tf.keras.optimizers.Adam(0.001))
    return model


def dnn_regression_wrapper(initial_data, target_data, model_order):
    print("Aloha! this is dnn_regression_wrapper!!!")
    # Make NumPy printouts easier to read.
    test_results = {}

    np.set_printoptions(precision=3, suppress=True)
    print(tf.__version__)

    initial_data = apply_log(initial_data)
    target_data = apply_log(target_data)

    initial_data_train, initial_data_test, target_data_train, target_data_test = splitting_wrapper(initial_data,
                                                                                                   target_data)

    dict_train, _, _ = convert_for_n_order_modelling(initial_data_train, target_data_train, model_order)
    dict_test, u_test, y_test = convert_for_n_order_modelling(initial_data_test, target_data_test, model_order)

    #dict_train, _, _ = convert_for_n_order_modelling_upsampling(initial_data_train, target_data_train, model_order)
    #dict_test, u_test, y_test = convert_for_n_order_modelling_upsampling(initial_data_test, target_data_test, model_order)

    data_train = alternative_formatting_for_modelling(dict_train, model_order)
    data_test = alternative_formatting_for_modelling(dict_test, model_order)

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

