from apple_chess import Board
from model_path_management import *
import tensorflow as tf
import tensorflow.keras as K
import numpy as np
import os
from datetime import datetime
import time
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

NO_OF_GAMES = 10_000
SAVE_DATA_EVERY = 50
UPDATE_MODEL_EVERY = 50
MAX_TRAINING_SIZE = 500_000


def add_cov_block(nn):
    nn.add(K.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding="same"))
    nn.add(K.layers.BatchNormalization())
    nn.add(K.layers.Activation(tf.nn.relu))


def load_model(dir):
    try:
        print("{} loaded!".format(MODEL_NAME))
        return K.models.load_model(dir)
    except:
        time.sleep(2)
        print("{} loaded!".format(MODEL_NAME))
        return K.models.load_model(dir)


if os.path.exists(get_last_model_dir(best_model_path)):
    model = load_model(get_last_model_dir(best_model_path))
elif os.path.exists(get_last_model_dir(model_path)):
    print("Model created but cannot find best model")
    exit()
else:  # create model
    model = K.Sequential()
    model.add(K.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding="same", input_shape=(8, 8, 2)))
    model.add(K.layers.BatchNormalization())
    model.add(K.layers.Activation(tf.nn.relu))
    for i in range(10):
        add_cov_block(model)
    model.add(K.layers.Conv2D(filters=1, kernel_size=(1, 1), strides=1, padding="same"))
    model.add(K.layers.BatchNormalization())
    model.add(K.layers.Flatten())
    model.add(K.layers.Dense(128))
    model.add(K.layers.Activation(tf.nn.relu))
    model.add(K.layers.Dense(1))
    model.add(K.layers.Activation(tf.nn.tanh))
    print("model created!")
    model.save(get_next_model_dir(model_path))
    model.save(get_next_model_dir(best_model_path))


def value_func(board, nn):
    value = np.mean(nn(np.array(board.as_nn_input()), training=False).numpy())
    return value


outcome = []

if os.path.exists(training_data_dir + "_0.npz"):
    training_data = np.load(training_data_dir + "_0.npz")
    training_data_x = training_data["x_train"].tolist()
    training_data_y = training_data["y_train"].tolist()
else:
    training_data_x = []
    training_data_y = []
    print("Couldn't find training data. Creating new")

for i in range(1, NO_OF_GAMES + 1):
    # create data
    start = datetime.now()
    B = Board(train_random=True)
    temp_x_s = B.as_nn_input()

    # play games
    while not B.is_terminal():
        move = B.alpha_beta_value(value_func, model, 1, -1e2, 1e2, max_depth=1, epsilon=0.04)[1]
        B.add(B.turn, move[0], move[1])
        temp_x_s = temp_x_s + B.as_nn_input()

    winner = B.winner()
    temp_y_s = [np.array([winner]) for _ in range(len(temp_x_s))]
    training_data_x = training_data_x + temp_x_s
    training_data_y = training_data_y + temp_y_s

    outcome.append(winner)

    if len(training_data_x) != len(training_data_y):
        print("The length of x and y data don't match")

    if i % SAVE_DATA_EVERY == 0:
        if len(training_data_x) > MAX_TRAINING_SIZE:
            training_data_x = training_data_x[-MAX_TRAINING_SIZE:]
        if len(training_data_y) > MAX_TRAINING_SIZE:
            training_data_y = training_data_y[-MAX_TRAINING_SIZE:]
        np.savez(training_data_dir + "_0",
                 x_train=np.array(training_data_x, dtype=np.float32),
                 y_train=np.array(training_data_y, dtype=np.float32))
        print("{} saved.".format(training_data_dir + "_0"))
        np.savez(training_data_dir + "_1",
                 x_train=np.array(training_data_x, dtype=np.float32),
                 y_train=np.array(training_data_y, dtype=np.float32))
        print("{} saved.".format(training_data_dir + "_1"))

    if i % UPDATE_MODEL_EVERY == 0:
        model = load_model(get_last_model_dir(best_model_path))

    time_taken = datetime.now() - start
    print("Winner: {:+}, Game length: {}, Game no: {}, Run time: {}".format(winner,
                                                                            int(len(temp_x_s) / 4),
                                                                            i,
                                                                            time_taken))

print("completed!")
print("Outcomes: {}".format(outcome))
