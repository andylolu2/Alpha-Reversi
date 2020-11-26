from constants import BOARD_DIM, HISTORY_LEN, MAX_TRAINING_SIZE, NO_OF_GAMES, NO_OF_RES_BLOCKS, SAVE_DATA_EVERY, UPDATE_MODEL_EVERY
from apple_chess import Board
from model_path_management import MODEL_NAME, best_model_path, get_last_model_dir, model_path, get_next_model_dir, training_data_dir
import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import pathlib
from datetime import datetime
import time
import multiprocessing

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def conv_block(x):
    x = keras.layers.Conv2D(filters=256, kernel_size=3, strides=1, padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation(tf.keras.activations.relu)(x)
    return x

def res_block(x):
    _input = x
    x = conv_block(x)
    x = keras.layers.Conv2D(filters=256, kernel_size=3, strides=1, padding="same")(x)
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

def load_model(dir):
    try:
        print("{} loaded!".format(MODEL_NAME))
        return keras.models.load_model(dir)
    except Exception:
        time.sleep(3)
        print("{} loaded!".format(MODEL_NAME))
        return keras.models.load_model(dir)


if os.path.exists(get_last_model_dir(best_model_path)):
    model = load_model(get_last_model_dir(best_model_path))
else:  # create model
    inputs = keras.Input(shape=BOARD_DIM+(HISTORY_LEN * 2 + 1,), name="state")
    x = conv_block(inputs)
    for _ in range(NO_OF_RES_BLOCKS):
        x = res_block(x)
    policy = policy_head(x)
    value = value_head(x)
    model = keras.Model(inputs=inputs, outputs=[policy, value], name="alpha_reversi")
    print("model created!")

    # plot and save model
    keras.utils.plot_model(model, "model_architecture.png", show_shapes=True, dpi=200)
    compete_model_dir = pathlib.Path(get_next_model_dir(model_path))
    best_model_dir = pathlib.Path(get_next_model_dir(best_model_path))
    if not compete_model_dir.parent.is_dir():
        compete_model_dir.parent.mkdir(parents=True)
    if not best_model_dir.parent.is_dir():
        best_model_dir.parent.mkdir(parents=True)
    model.save(str(compete_model_dir))
    model.save(str(best_model_dir))


if os.path.exists(training_data_dir + "_0.npz"):
    data = np.load(training_data_dir + "_0.npz")
    data_state = data["state"].tolist()
    data_policy = data["policy"].tolist()
    data_value = data["value"].tolist()
else:
    data_state = []
    data_policy = []
    data_value = []
    print("Couldn't find training data. Creating new")

for i in range(1, NO_OF_GAMES + 1):
    # create data
    start = datetime.now()
    B = Board(train_random=True)
    temp_states = []
    temp_policies = []
    # play games
    while not B.is_terminal():
        temp_states.append(B.as_nn_input()[0])

        policy = B.mcts(model, iter=16)
        temp_policies.append(policy)

        move = B.get_mcts_move(policy)
        B = B.traverse(move[0], move[1])
        print(repr(B))

    winner = B.winner()
    temp_values = [[winner] for _ in range(len(temp_states))]

    data_state += temp_states
    data_policy += temp_policies
    data_value += temp_values

    assert len(data_state) == len(data_policy) == len(data_value), "The length of data does not match"

    if i % SAVE_DATA_EVERY == 0:
        if len(data_state) > MAX_TRAINING_SIZE:
            data_state = data_state[-MAX_TRAINING_SIZE:]
            data_policy = data_policy[-MAX_TRAINING_SIZE:]
            data_value = data_value[-MAX_TRAINING_SIZE:]
        np.savez(training_data_dir + "_0",
            data_state = np.array(data_state, dtype=np.float32),
            data_policy = np.array(data_policy, dtype=np.float32),
            data_value = np.array(data_value, dtype=np.float32))
        print(f"{training_data_dir}_0 saved.")
        np.savez(training_data_dir + "_1",
            data_state = np.array(data_state, dtype=np.float32),
            data_policy = np.array(data_policy, dtype=np.float32),
            data_value = np.array(data_value, dtype=np.float32))
        print(f"{training_data_dir}_1 saved.")

    if i % UPDATE_MODEL_EVERY == 0:
        model = load_model(get_last_model_dir(best_model_path))

    print(f"Winner: {winner:+}, Game length: {len(temp_states)}, Game no: {i}, Run time: {datetime.now() - start}")

print("completed!")
