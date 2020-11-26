from constants import C_ELO, NO_OF_GAMES, WINRATE_CUTOFF
from apple_chess import Board
from model_path_management import *
import numpy as np
import os
import math
from datetime import datetime
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import Session
import multiprocessing as mp


def compete_with_best_model():
    try:
        name = mp.current_process().name
        print(name, "new process started", flush=True)
        import tensorflow.keras as K

        config = ConfigProto()
        config.gpu_options.allow_growth = True
        session = Session(config=config)

        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

        if os.path.exists(get_elo_rating_dir(model_elo_rating_path)):
            elo_ratings = np.load(get_elo_rating_dir(model_elo_rating_path))
            models_elo_rating = elo_ratings["models"].tolist()
            best_models_elo_rating = elo_ratings["best_models"].tolist()
        else:
            models_elo_rating = [100]
            best_models_elo_rating = [100]
            
        if os.path.exists(get_last_model_dir(model_path)):
            compete_model = K.models.load_model(get_last_model_dir(model_path))
            print(name, "Compete model {} version {} loaded!".format(MODEL_NAME, get_last_model_index(model_path)))
            if os.path.exists(get_last_model_dir(best_model_path)):
                best_model = K.models.load_model(get_last_model_dir(best_model_path))
                print(name, "Best model loaded!")

                best_model_wins = 0

                for i in range(NO_OF_GAMES):
                    # create data
                    start = datetime.now()
                    B = Board(train_random=True)

                    # play games
                    while not B.is_terminal():
                        if B.turn == (-1) ** i:
                            move = B.alpha_beta_value(best_model, 1, epsilon=0.015)[1]
                            B = B.traverse(move[0], move[1])
                        else:
                            move = B.alpha_beta_value(compete_model, 1, epsilon=0.015)[1]
                            B = B.traverse(move[0], move[1])

                    winner = B.winner()
                    if winner == (-1) ** i:
                        best_model_wins += 1
                        winner_model = "Best model"
                    else:
                        winner_model = "Compete model"

                    time_taken = datetime.now() - start
                    print(name, "Winner: {}, Game no: {}, Run time: {}".format(winner_model, i, time_taken))

                best_model_win_rate = float(best_model_wins / NO_OF_GAMES)

                # Prevent zero division errors
                if best_model_win_rate == 0:
                    best_model_win_rate += 1e-8
                elif best_model_win_rate == 1:
                    best_model_win_rate -= 1e-8

                # Calculate elo of new model
                best_model_elo = best_models_elo_rating[-1]
                compete_model_elo = (math.log(1 / best_model_win_rate - 1, 10) / C_ELO) + best_model_elo
                models_elo_rating.append(compete_model_elo)

                if best_model_win_rate < 1 - WINRATE_CUTOFF:  # New model is significantly better
                    best_models_elo_rating.append(compete_model_elo)
                    compete_model.save(get_next_model_dir(best_model_path), save_format="tf")
                    print(name, "Compete model won with {:.2%} win rate".format(1 - best_model_win_rate))
                else:  # Current best is still probably better
                    print(name, "Best model won with {:.2%} win rate".format(best_model_win_rate))
                if not os.path.exists("model_elo_rating\\{}".format(MODEL_NAME)):
                    os.makedirs("model_elo_rating\\{}".format(MODEL_NAME))
                np.savez(get_elo_rating_dir(model_elo_rating_path),
                         models=np.array(models_elo_rating),
                         best_models=np.array(best_models_elo_rating))
        return
    except Exception as e:
        print(repr(e), flush=True)
