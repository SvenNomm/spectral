# this file contains the functions necessary for the transformers support
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from preprocessing_module import splitting_wrapper
import tensorflow.compat.v1.keras.backend as K


#tf.compat.v1.disable_eager_execution()



def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Normalization and Attention
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res


def transformer_decoder(outputs, head_size, num_heads, ff_dim, dropout=0):
    # Normalization and Attention
    y = layers.LayerNormalization(epsilon=1e-6)(outputs)
    y = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(y, y)
    y = layers.Dropout(dropout)(y)
    res = y + outputs

    # Feed Forward Part
    y = layers.LayerNormalization(epsilon=1e-6)(res)
    y = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(y)
    y = layers.Dropout(dropout)(y)
    y = layers.Conv1D(filters=outputs.shape[-1], kernel_size=1)(y)
    return y + res


def positional_encoding(max_position, d_model, min_freq=1e-4):
    position = tf.range(max_position, dtype=tf.float32)
    mask = tf.range(d_model)
    sin_mask = tf.cast(mask%2, tf.float32)
    cos_mask = 1-sin_mask
    exponent = 2*(mask//2)
    exponent = tf.cast(exponent, tf.float32)/tf.cast(d_model, tf.float32)
    freqs = min_freq**exponent
    angles = tf.einsum('i,j->ij', position, freqs)
    pos_enc = tf.math.cos(angles)*cos_mask + tf.math.sin(angles)*sin_mask
    return pos_enc


def build_model(
    input_shape,
    output_shape,
    head_size,
    num_heads,
    ff_dim,
    num_transformer_blocks,
    mlp_units,
    dropout=0,
    mlp_dropout=0,
):
    encoder_inputs = keras.Input(shape=input_shape)

    outputs = keras.Input(shape=output_shape)
    x = encoder_inputs
    y = outputs

    x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)
    y = transformer_decoder(y, head_size, num_heads, ff_dim, dropout)

    #x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)

        y = layers.Dense(dim, activation="relu")(y)
        y = layers.Dropout(mlp_dropout)(y)
    decoder = keras.Model([x], y)
    decoder_outputs = decoder([x,y])
    transformer = keras.Model(
        [x, y], decoder_outputs, name="transformer"
    )
    return keras.Model(encoder_inputs, outputs)
