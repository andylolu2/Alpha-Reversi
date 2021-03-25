from constants import BOARD_DIM, HISTORY_LEN, NO_OF_FILTERS, NO_OF_RES_BLOCKS
from tensorflow import keras
import tensorflow as tf

def conv_block(x):
    x = keras.layers.Conv2D(filters=NO_OF_FILTERS, kernel_size=3, strides=1, padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation(tf.keras.activations.relu)(x)
    return x

def res_block(x):
    _input = x
    x = conv_block(x)
    x = keras.layers.Conv2D(filters=NO_OF_FILTERS, kernel_size=3, strides=1, padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.add([x, _input])
    x = keras.layers.Activation(tf.keras.activations.relu)(x)
    return x

def policy_head(x):
    x = keras.layers.Conv2D(filters=2, kernel_size=1, strides=1, padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation(tf.keras.activations.relu)(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(BOARD_DIM[0]*BOARD_DIM[1], name="policy")(x)
    return x

def value_head(x):
    x = keras.layers.Conv2D(filters=1, kernel_size=1, strides=1, padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation(tf.keras.activations.relu)(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(256)(x)
    x = keras.layers.Activation(tf.keras.activations.relu)(x)
    x = keras.layers.Dense(1)(x)
    x = keras.layers.Activation(tf.keras.activations.tanh, name="value")(x)
    return x

def create_model():
    inputs = keras.Input(shape=BOARD_DIM+(HISTORY_LEN * 2 + 1,), name="state")
    x = conv_block(inputs)
    for _ in range(NO_OF_RES_BLOCKS):
        x = res_block(x)
    policy = policy_head(x)
    value = value_head(x)
    return keras.Model(inputs=inputs, outputs=[policy, value], name="alpha_reversi")