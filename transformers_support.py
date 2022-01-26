# this file contains the functions necessary for the transformers support
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from preprocessing_module import splitting_wrapper
from transformer_1 import build_model
#def tokenize_data(data):
from language_based_transformer import preprocess_sequences


def transformer_workflow(initial_data, target_data):
    preprocess_sequences(initial_data, target_data)



    initial_data_train, initial_data_test, target_data_train, target_data_test = \
        splitting_wrapper(initial_data, target_data)

    initial_data_train = initial_data_train.to_numpy()
    initial_data_test = initial_data_test.to_numpy()
    target_data_train = target_data_train.to_numpy()
    target_data_test = target_data_test.to_numpy()


    x_train = initial_data_train.reshape((initial_data_train.shape[0], initial_data_train.shape[1], 1))
    x_test = initial_data_test.reshape((initial_data_test.shape[0], initial_data_test.shape[1], 1))

    y_train = target_data_train.reshape((target_data_train.shape[0], target_data_train.shape[1], 1))
    y_test = target_data_test.reshape((target_data_test.shape[0], target_data_test.shape[1], 1))

    # for now I will skip this line
    #idx = np.random.permutation(len(x_train))
    #x_train = x_train[idx]
    #y_train = y_train[idx]

    # most probably in this case empty row should be added
    x_train[y_train == -1] = 0
    x_test[y_test == -1] = 0

    y_train[y_train == -1] = 0
    y_test[y_test == -1] = 0

    embed_dim = 256
    latent_dim = 2048
    num_heads = 8

    input_shape = x_train.shape[1:]
    output_shape = y_train.shape[1:]
    dropout = 0,
    mlp_dropout = 0,

    model = build_model(
        input_shape,
        output_shape,
        head_size=256,
        num_heads=4,
        ff_dim=4,
        num_transformer_blocks=4,
        mlp_units=[128],
        mlp_dropout=0,
        dropout=0.25,)

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        metrics=["sparse_categorical_accuracy"],
    )
    model.summary()

    callbacks = [keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)]

    model.fit(
        x_train,
        y_train,
        validation_split=0.2,
        epochs=200,
        batch_size=64,
        callbacks=callbacks,
    )

    model.evaluate(x_test, y_test, verbose=1)

    print("That's all folks!!!")
    return y_test
