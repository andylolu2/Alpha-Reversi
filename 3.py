from apple_chess import Board
from model_path_management import *
import os
from datetime import datetime
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import Session
import tensorflow.keras as K


def value_func(board, nn):
    import numpy as np
    value = np.mean(nn(np.array(board.as_nn_input()), training=False).numpy())
    return value


def compete(model_1, model_2):
    try:

        config = ConfigProto()
        config.gpu_options.allow_growth = True
        session = Session(config=config)

        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

        NO_OF_GAMES = 20

        model_1_wins = 0

        for i in range(1, NO_OF_GAMES + 1):

            start = datetime.now()
            B = Board(train_random=True)

            # play games
            while not B.is_terminal():
                if B.turn == (-1) ** i:
                    move = B.alpha_beta_value(value_func, model_1, 2, -1e2, 1e2, max_depth=2, epsilon=0.015)[1]
                    B.add(B.turn, move[0], move[1])
                elif B.turn == (-1) ** (i + 1):
                    move = B.alpha_beta_value(value_func, model_2, 2, -1e2, 1e2, max_depth=2, epsilon=0.015)[1]
                    B.add(B.turn, move[0], move[1])

            winner = B.winner()
            if winner == (-1) ** i:
                model_1_wins += 1
                winner_model = "Model 1"
            else:
                winner_model = "Model 2"

            time_taken = datetime.now() - start
            print("Winner: {}, Game no: {}, Model 1 : Model 2 = {} : {} Run time: {}".format(winner_model, i,
                                                                                             model_1_wins,
                                                                                             i - model_1_wins,
                                                                                             time_taken))

        model_1_win_rate = float(model_1_wins / NO_OF_GAMES)
        if model_1_win_rate == 0:
            model_1_win_rate += 1e-8
        elif model_1_win_rate == 1:
            model_1_win_rate -= 1e-8
        if model_1_win_rate < 0.5:
            print("Model 2 won with {:.2%} win rate".format(1 - model_1_win_rate))
        else:
            print("Model 1 won with {:.2%} win rate".format(model_1_win_rate))
    except Exception as e:
        print(repr(e), flush=True)


bad_model = K.models.load_model(model_path.format(MODEL_NAME, MODEL_NAME, 10))
good_model = K.models.load_model(get_last_model_dir(best_model_path))
compete(model_1=good_model, model_2=bad_model)