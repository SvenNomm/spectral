# this files contains functions performing basic fitting attempt


import pandas as pd
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LinearRegression
from scipy.signal import resample
from sklearn.neural_network import MLPRegressor
from preprocessing_module import splitting_wrapper
from preprocessing_module import convert_for_n_order_modelling
from preprocessing_module import convert_for_n_order_modelling_upsampling
from preprocessing_module import alternative_formatting_for_modelling
from preprocessing_module import apply_log
import matplotlib.pyplot as plt


def fit_regression_raw_by_raw(initial_data, target_data):
    print("Aloha! this is fit regression row by row function!")
    rows = len(target_data)

    # the following two lines are very local
    del initial_data['k']
    del target_data['f']

    target_columns = target_data.columns
    initial_columns = initial_data.columns

    initial_columns = initial_columns[0: len(target_columns)]

    for i in range(1, rows-1):
        print("------------------------------------------------")
        x = initial_data.loc[i - 1, :].to_numpy()
        #x = initial_data.loc[i - 1, initial_columns].to_numpy()
        l = len(target_columns)
        x = resample(x, num=len(target_columns))
        y = target_data.loc[i, :].to_numpy()

        fig, axis = plt.subplots()
        plt.plot(x,'blue')
        plt.plot(y,'green')
        plt.show
        #reg = LinearRegression()
        x = x.reshape(-1,1)
        #reg.fit(x,y)
        #print(reg.score(x,y))
        #print(reg.coef_)
        #print(reg.intercept_)
        #print("________________________________________________")
        #print("--------- Perceptron ---------------------------")
        if i > 2:
            y_ampl = np.abs(np.max(y) - np.min(y))

            y_hat = clf.predict(x)
            residuals = (y - y_hat) / y_ampl
            fig1, axis = plt.subplots()
            plt.plot(residuals)
            plt.show()

        clf = MLPRegressor(hidden_layer_sizes=(10,5,), random_state=1, max_iter=355, warm_start=True)
        clf.fit(x,y)
        y_hat = clf.predict(x)
        y_ampl = np.abs(np.max(y) - np.min(y))
        residuals = (y - y_hat) / y_ampl
        fig1, axis = plt.subplots()
        plt.plot(residuals)
        plt.show()


def model_training_wrapper(initial_data, target_data, model_order):
    initial_data_train, initial_data_test, target_data_train, target_data_test = splitting_wrapper(initial_data,
                                                                                                   target_data)

    _, u_train, y_train = convert_for_n_order_modelling(initial_data_train, target_data_train, model_order)
    dict_test, u_test, y_test = convert_for_n_order_modelling(initial_data_test, target_data_test, model_order)

    # lest feet linear regression
    reg = LinearRegression()
    print(np.argwhere(np.isnan(u_train)))
    print(np.argwhere(np.isnan(y_train)))
    reg.fit(u_train, y_train)
    print(reg.score(u_train, y_train))
    print(reg.coef_)
    print(reg.intercept_)
    clf = MLPRegressor(hidden_layer_sizes=(5, 3,), random_state=1, max_iter=355, warm_start=True)
    clf.fit(u_train, y_train)

    print("Performing validation of the linear regression model.")
    dict_test_keys = dict_test.keys()
    for key in dict_test_keys:
        data_array = dict_test[key]
        u_test = data_array[:, 0:model_order-1]
        y_test = data_array[:, model_order]
        y_hat = reg.predict(u_test)
        y_ampl = np.abs(np.max(y_test) - np.min(y_test))
        residuals = (y_test - y_hat) / y_ampl
        fig1, axis = plt.subplots()
        plt.plot(residuals)
        plt.title('regression')
        plt.show()
        fig2, axis = plt.subplots()
        plt.plot(y_test)
        plt.plot(y_hat)
        plt.title('regression')
        plt.show()


        y_hat = clf.predict(u_test)
        y_ampl = np.abs(np.max(y_test) - np.min(y_test))
        residuals = (y_test - y_hat) / y_ampl
        fig1, axis = plt.subplots()
        plt.plot(residuals)
        plt.title('NN')
        plt.show()

        fig2, axis = plt.subplots()
        plt.plot(y_test)
        plt.plot(y_hat)
        plt.title('regression')
        plt.show()


def model_training_wrapper_2(initial_data, target_data, model_order):
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

    reg = LinearRegression()
    print(np.argwhere(np.isnan(data_train)))
    print(np.argwhere(np.isnan(data_test)))
    reg.fit(data_train[:, 0:model_order-1], data_train[:, model_order])
    print(reg.score(data_train[:, 0:model_order-1], data_train[:, model_order]))
    print(reg.coef_)
    print(reg.intercept_)

    y_hat = reg.predict(data_test[:, 0:model_order-1])
    y_ampl = np.abs(np.max(data_test[:, model_order]) - np.min(data_test[:, model_order]))
    residuals = (data_test[:, model_order] - y_hat) / y_ampl

    #clf = MLPRegressor(hidden_layer_sizes=(55,15,), random_state=1, max_iter=355, warm_start=True) # for low resolution
    clf = MLPRegressor(hidden_layer_sizes=(55, 15), random_state=1, max_iter=355,
                       warm_start=True)  # for high resolution
    clf.fit(data_train[:, 0:model_order-1], data_train[:, model_order])

    y_hat_nn = clf.predict(data_test[:, 0:model_order-1])
    residuals_nn = (data_test[:, model_order] - y_hat_nn) / y_ampl

    fig1, axis = plt.subplots()
    plt.plot(data_test[:, model_order], color='blue')
    plt.plot(y_hat, color='orange')
    plt.plot(y_hat_nn, color='green')
    plt.title("actual signals")
    plt.show()

    fig2, axis = plt.subplots()
    plt.plot(residuals, color='orange')
    plt.plot(residuals_nn, color='green')
    plt.title("residuals")
    plt.show()

    test_keys = dict_test.keys()
    for key in test_keys:
        data_test = dict_test[key]
        y_ampl = np.abs(np.max(data_test[:, model_order]) - np.min(data_test[:, model_order]))
        y_hat_nn = clf.predict(data_test[:, 0:model_order - 1])
        residuals_nn = (data_test[:, model_order] - y_hat_nn) / y_ampl

        y_hat = reg.predict(data_test[:, 0:model_order - 1])
        residuals = (data_test[:, model_order] - y_hat) / y_ampl

        fig3, axis = plt.subplots()
        plt.plot(data_test[:, model_order], color='blue')
        plt.plot(y_hat_nn, color='green')
        plt.plot(y_hat, color='orange')
        plt.title("validation on a small set")
        plt.show()

        #fig3, axis = plt.subplots()
        #plt.plot(data_test[:, model_order], color='blue')
        #plt.plot(residuals_nn, color='green')
        #plt.plot(residuals, color='orange')
        #plt.title("residuals for a small set")
        #plt.show()

