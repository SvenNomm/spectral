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
from transformer_1 import positional_encoding
from transformer_1 import transformer_decoder
from transformer_1 import transformer_encoder

def transformer_workflow_x(initial_data, target_data):
    preprocess_sequences(initial_data, target_data)

    initial_data_train, initial_data_test, target_data_train, target_data_test, test_index = \
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
    latent_dim = 100 #2048
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


def format_dataset(x, y):
    return({"encoder_inputs": x, "decoder_inputs": y,}, y)


def data_for_transformer(x, y, batch_size):
    rows = len(x)
    xx = []
    yy = []
    for i in range(0, rows):
        xx.append(x[i, :])
        yy.append(y[i, :])

    dataset = tf.data.Dataset.from_tensor_slices((xx, yy))
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(format_dataset)
    return dataset


def format_dataset_val (x, y):
    return({"encoder_inputs": x, "decoder_inputs": y,})


def validation_data(x, y, batch_size):
    rows = len(x)
    xx =[]
    yy =[]
    for i in range(0, rows):
        xx.append(x[i, :])
        yy.append(y[i, :])

    dataset_x = tf.convert_to_tensor(xx)
    dataset_y = tf.convert_to_tensor(yy)
    #dataset = tf.data.Dataset.from_tensor_slices((xx))
    #dataset = dataset.batch(batch_size)
    #dataset = dataset.map(format_dataset_val)
    return dataset_x, dataset_y


def test_model(test_x, test_y, model, test_index):
    print("Testing LSTM model!")
    #test_x = apply_log(test_x)
    #test_y = apply_log(test_y)
    test_x = test_x.to_numpy()
    test_y = test_y.to_numpy()

    rows, cols = test_y.shape
    test_x = test_x.reshape(rows, cols, 1)


    y_hat = model.predict(test_x, batch_size=2, verbose=0)
    for i in range(0, rows):
        #x_hat = test_x[i, :, :]
        #x_hat = x_hat[None, :]
        #y_hat = model.predict(test_x[i, :, :])
        print("Testing for datapoint", test_index.loc[i])
        y_ampl = np.abs(np.max(test_y[i, :]) - np.min(test_y[i, :]))
        residuals_nn = (test_y[i, :] - y_hat[i, :]) / y_ampl

        fig2, axis = plt.subplots()
        plt.plot(test_y[i, :], color='blue')
        plt.plot(y_hat[i, :], color='orange')
        #plt.title("validation for", str(test_index.loc[i]))
        plt.show()

        fig3, axis = plt.subplots()
        plt.plot(residuals_nn, color='green')
        plt.title("residuals for a small set")
        plt.show()
    print("LSTM model has been tested! ")


def transformer_workflow(initial_data, target_data):
    preprocess_sequences(initial_data, target_data)

    initial_data_train, initial_data_test, target_data_train, target_data_test, test_index = \
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
    #x_train[y_train == -1] = 0
    #x_test[y_test == -1] = 0

    #y_train[y_train == -1] = 0
    #y_test[y_test == -1] = 0

    train_ds = data_for_transformer(initial_data_train, target_data_train, batch_size=64)
    val_ds = data_for_transformer(initial_data_test, target_data_test,batch_size=64)

    #train_ds = data_for_transformer(x_train, y_train, batch_size=64)
    #val_ds = data_for_transformer(x_test, y_test,batch_size=64)


    latent_dim = 43 #2048
    num_heads = 8

    input_shape = x_train.shape[1:]
    output_shape = y_train.shape[1:]
    dropout = 0
    mlp_dropout = 0
    ff_dim = 4
    embed_dim = 43
    head_size=256

    encoder_inputs = keras.Input(shape=input_shape, dtype="float32", name="encoder_inputs")
    #encoder_inputs = keras.Input(shape=input_shape)
    x = positional_encoding(43, 43, min_freq=1e-4)
    x = x * encoder_inputs
    encoder_outputs = transformer_encoder(x, head_size, num_heads, ff_dim, dropout=0)
    encoder = keras.Model(encoder_inputs, encoder_outputs)
    #decoder_inputs = keras.Input(shape=output_shape)
    decoder_inputs = keras.Input(shape=output_shape, dtype="float32", name="decoder_inputs")
    encoded_seq_inputs = keras.Input(shape=(None, embed_dim), name="decoder_state_inputs")
    x = positional_encoding(43, 43, min_freq=1e-4)
    x = x * decoder_inputs
    x = transformer_decoder(x, head_size, num_heads, ff_dim, dropout)
    x = layers.Dropout(0.5)(x)
    decoder_outputs = layers.Dense(latent_dim,activation="relu")(x)
    decoder = keras.Model([decoder_inputs, encoded_seq_inputs], decoder_outputs)
    decoder_outputs = decoder([decoder_inputs, encoder_outputs])
    transformer = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs, name="transformer")

    epochs = 1  # This should be at least 30 for convergence

    transformer.summary()

    transformer.compile("rmsprop", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    transformer.fit(train_ds, epochs=epochs, validation_data=val_ds)

    #callbacks = [keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)]

    #for i in range(0, len())

    val_x, val_y = validation_data(initial_data_test, target_data_test, batch_size=64)
    a = transformer([val_x, val_y])
    a = a.numpy()
    print(a)

    print("That's all folks!!!")
    return y_test