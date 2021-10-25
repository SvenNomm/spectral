import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.models import load_model

from analyzer.settings import *


class LstmSupport:
    @staticmethod
    def build_model():
        model = Sequential()
        layers = {'input': 1, 'hidden1': 64, 'hidden2': 256, 'hidden3': 100, 'output': 1}

        model.add(LSTM(
            input_length=Settings.WINDOW_SIZE - 1,
            input_dim=layers['input'],
            output_dim=layers['hidden1'],
            return_sequences=True))

        model.add(Dropout(0.2))
        model.add(LSTM(layers['hidden2'], return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(layers['hidden3'], return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(output_dim=layers['output']))
        model.add(Activation("linear"))
        start = time.time()
        model.compile(loss="mse", optimizer="rmsprop", metrics=['accuracy'])

        print("Compilation Time : ", time.time() - start)
        return model

    @staticmethod
    def train_and_save(train_x, train_y, model_path):
        start_time = time.time()
        print('building model...')
        model = LstmSupport.build_model()
        print("training...")
        model.fit(
            train_x, train_y,
            batch_size=Settings.BATCH_SIZE,
            nb_epoch=Settings.EPOCHS,
            validation_split=0.05)
        print('prediction duration (s) : ', time.time() - start_time)
        print('saving model: %s' % model_path)
        model.save(model_path)
        return model

    @staticmethod
    def train_and_save_get_history(train_x, train_y, model_path):
        start_time = time.time()
        print('building model...')
        model = LstmSupport.build_model()
        print("training...")
        history = model.fit(
            train_x, train_y,
            batch_size=Settings.BATCH_SIZE,
            nb_epoch=Settings.EPOCHS,
            validation_split=0.05)
        print('prediction duration (s) : ', time.time() - start_time)
        print('saving model: %s' % model_path)
        model.save(model_path)
        return model, history

    @staticmethod
    def load_model(path):
        return load_model(filepath=path)

    @staticmethod
    def predict(test_x, model=None, model_path=None):

        if model is None:
            if model_path is not None:
                model = load_model(model_path)
                print('model %s was loaded', model_path)
            else:
                print('model and model_path are missing...')
                return None