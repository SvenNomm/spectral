# this file contains the functions supporting deep learning module

import time
from keras.layers.core import Dense, Activation, Dropout
#from keras.layers.recurrent import LSTM
from keras.layers import LSTM
from keras.models import Sequential
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from preprocessing_module import apply_log
from preprocessing_module import apply_normalization
from sklearn.preprocessing import normalize


class LstmSettings:
    EPOCHS = 10
    WINDOW_SIZE = 20
    BATCH_SIZE = 10


def build_model(train_x, train_y):  # NB most probably faulty example to be deleted
    input_shape = train_x.shape
    output_shape = train_y.shape

    model = Sequential()
    #layers = {'input': 1, 'hidden1': 64, 'hidden2': 256, 'hidden3': 100, 'output': 1}

    model.add(LSTM(
        units=50,
        input_shape=train_x.shape))
        #output_dim=len(train_y)))
    #    return_sequences=True))

    #lstm = LSTM(4)
    #model.add(lstm)

    #model.add(Dropout(0.2))
    #model.add(LSTM(layers['hidden2'], return_sequences=True))
    #model.add(Dropout(0.2))
    #model.add(LSTM(layers['hidden3'], return_sequences=False))
    #model.add(Dropout(0.2))
    #model.add(Dense(output_dim=layers['output']))
    #model.add(Dense(layers['output']))
    #model.add(Activation("linear"))
    start = time.time()
    model.compile(loss="mse", optimizer="rmsprop", metrics=['accuracy'])

    print("Compilation Time : ", time.time() - start)
    return model


def build_lstm_model(train_x, train_y):
    print("Building LSTM model!")
    rows, cols = train_x.shape
    model = Sequential()
    model.add(LSTM(8, input_shape=(cols, 1), return_sequences=True))   # 64, 64, 64
    model.add(LSTM(8))
    model.add(Dense(64))
    model.compile(loss='mean_squared_error', optimizer='adam')
    print("LSTM model is ready")
    model.summary()
    #
    return model



def train_model(train_x, train_y):
    print("Training LSTM model!")
    start_time = time.time()
    model = build_lstm_model(train_x, train_y)
    train_x = apply_log(train_x)
    train_y = apply_log(train_y)

    train_x = train_x.to_numpy()
    train_y = train_y.to_numpy()

    #train_x = apply_normalization(train_x)
    #train_y = apply_normalization(train_y)

    rows, cols = train_x.shape
    train_x = train_x.reshape(rows, cols, 1)
    print("training...")
    model.fit(train_x, train_y,
        epochs=1000,
        batch_size=28,
        verbose=2)
    #for i in range(0, rows):
    #    x = train_x[i, :].reshape(1, cols, 1)
    #    model.fit(
    #        x, train_y[i, :],
    #        epochs=1,
    #        batch_size=1,
    #        verbose=2)
            #nb_epoch=LstmSettings.EPOCHS,
            #validation_split=0.05)
        #print('prediction duration (s) : ', time.time() - start_time)
        #print('saving model: %s' % model_path)
        #model.save(model_path)
    print("LSTM model has been trained!")
    return model


def test_model(test_x, test_y, model):
    print("Testing LSTM model!")
    test_x = apply_log(test_x)
    test_y = apply_log(test_y)
    test_x = test_x.to_numpy()
    test_y = test_y.to_numpy()

    rows, cols = test_y.shape
    test_x = test_x.reshape(rows, cols, 1)


    y_hat = model.predict(test_x, batch_size=2, verbose=0)
    for i in range(0, rows):
        #x_hat = test_x[i, :, :]
        #x_hat = x_hat[None, :]
        #y_hat = model.predict(test_x[i, :, :])

        y_ampl = np.abs(np.max(test_y[i, :]) - np.min(test_y[i, :]))
        residuals_nn = (test_y[i, :] - y_hat[i, :]) / y_ampl

        fig2, axis = plt.subplots()
        plt.plot(test_y[i, :], color='blue')
        plt.plot(y_hat[i, :], color='orange')
        plt.title("validation on a small set")
        plt.show()

        fig3, axis = plt.subplots()
        plt.plot(residuals_nn, color='green')
        plt.title("residuals for a small set")
        plt.show()
    print("LSTM model has been tested! ")



