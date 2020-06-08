from apple_chess import Board, DEPTH
from tqdm import tqdm
import tensorflow as tf
import tensorflow.keras as K
import numpy as np
import os
from datetime import datetime

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

NO_OF_GAMES = 10_000
TRAIN_EVERY = 20
SAVE_FREQ = 10
LEARNING_RATE = 1e-5
MOMENTUM = 0.9

MODEL_NAME = "alphago0"
model_dir = "models\\{}.tf".format(MODEL_NAME)


def add_conv_block(nn):
    nn.add(K.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding="same"))
    nn.add(K.layers.BatchNormalization())
    nn.add(K.layers.Activation(tf.nn.relu))


if os.path.exists(model_dir):
    main = K.models.load_model(model_dir)
    print("{} loaded!".format(MODEL_NAME))
else:
    # create model
    main = K.Sequential()
    main.add(K.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding="same", input_shape=(8, 8, 2)))
    main.add(K.layers.BatchNormalization())
    main.add(K.layers.Activation(tf.nn.relu))

    for i in range(9):
        add_conv_block(main)
    main.summary()

    policy_net = K.Sequential()
    policy_net.add(K.layers.Conv2D(filters=2, kernel_size=(1, 1), strides=1, padding="same", input_shape=(8, 8, 128)))
    policy_net.add(K.layers.BatchNormalization())
    policy_net.add(K.layers.Activation(tf.nn.relu))
    policy_net.add(K.layers.Flatten())
    policy_net.add(K.layers.Dense(64))
    policy_net.add(K.layers.Activation(tf.nn.softmax))

    value_net = K.Sequential()
    value_net.add(K.layers.Conv2D(filters=1, kernel_size=(1,1), strides=1, padding="same", input_shape=(8,8,128)))
    value_net.add(K.layers.BatchNormalization())
    value_net.add(K.layers.Activation(tf.nn.relu))
    value_net.add(K.layers.Flatten())
    value_net.add(K.layers.Dense(256))
    value_net.add(K.layers.Activation(tf.nn.relu))
    value_net.add(K.layers.Dense(1))
    value_net.add(K.layers.Activation(tf.nn.tanh))

    policy_net.summary()
    value_net.summary()
    # print("model created!")
    # model.save(model_dir)


# def value_func(board, nn):
#     return nn(np.array([board], dtype=np.float32), training=False).numpy()[0][0]
#
#
# progress_check = [[], []]
# outcome = []
# empty_board = []
# full_board = []
#
# for i in range(8):
#     full_board.append([])
#     for j in range(8):
#         full_board[i].append([0])
# index = 0
# for row in full_board:
#     for col in row:
#         if index <= 40:
#             col[0] = 1
#             index += 1
#         else:
#             col[0] = -1
#
# for i in range(8):
#     empty_board.append([])
#     for j in range(8):
#         empty_board[i].append([0])
# empty_board = np.array(empty_board)
# empty_board[3][4] = [1]
# empty_board[3][4] = [-1]
# empty_board[4][3] = [-1]
# empty_board[3][3] = [1]
#
# GAME_INDEX = 0
# TRAIN_INDEX = 0
#
# for i in range(int(NO_OF_GAMES / TRAIN_EVERY)):
#     x_s = []
#     y_s = []
#
#     for j in (range(TRAIN_EVERY)):
#         # create data
#         start = datetime.now()
#         B = Board()
#         temp_x_s = [np.copy(B.board)]
#
#         # play games
#         while not B.is_terminal():
#             move = B.alpha_beta_value(value_func, model, DEPTH, -1e2, 1e2)[1]
#             B.add(B.turn, move[0], move[1])
#             temp_x_s.append(np.copy(B.board))
#
#         for x in temp_x_s:
#             x_s.append(x)
#         winner = B.winner()
#         for k in range(len(temp_x_s)):
#             y_s.append([(winner - 0.5) * (k / (len(temp_x_s) - 1)) + 0.5])
#
#         GAME_INDEX += 1
#         outcome.append(winner)
#         time_taken = datetime.now() - start
#
#         print("Winner: {}, Game length: {}, Game no: {}, Run time: {}".format(winner, len(temp_x_s), GAME_INDEX,
#                                                                               time_taken))
#
#     # process data
#     training_data = []
#     for k in range(len(x_s)):
#         training_data.append([x_s[k], y_s[k]])
#     training_data = np.array(training_data)
#     np.random.shuffle(training_data)
#     for k in range(len(x_s)):
#         x_s[k] = np.copy(training_data[k][0])
#         y_s[k] = np.copy(training_data[k][1])
#     x_s = np.array(x_s)
#     y_s = np.array(y_s)
#
#     # train
#     # log_dir = "logs\\fit\\{}".format(0)
#     # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
#     model.fit(x=x_s,
#               y=y_s,
#               batch_size=64,
#               epochs=2,
#               verbose=2,
#               validation_split=0.05)
#     # callbacks=[tensorboard_callback]
#     TRAIN_INDEX += 1
#     if TRAIN_INDEX % SAVE_FREQ == 0:
#         model.save(model_dir, save_format="tf")
#         print("Model saved")
#     progress_check[0].append(value_func(empty_board))
#     progress_check[1].append(value_func(full_board))
#
# print("completed!")
# print(progress_check)
# print("Outcomes: {}".format(outcome))
