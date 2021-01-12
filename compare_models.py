from constants import COMPARATOR_C_PUCT, COMPARATOR_MCTS_ITERS, COMPARATOR_NO_OF_GAMES, C_ELO, MCTS_BATCH_SIZE, V_RESIGN, WINRATE_CUTOFF
from apple_chess import Board
from model_path_management import get_last_model_dir, get_elo_rating_dir, get_next_model_dir, get_last_model_dir, _get_last_model_index, model_elo_rating_path, model_path, best_model_path, MODEL_NAME
from helper_methods import load_model
import numpy as np
import os
import math
from datetime import datetime
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor


def compete_with_best_model():
    name = mp.current_process().name
    print(name, "new process started", flush=True)
    import tensorflow.keras as K
    import tensorflow as tf

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # load / create elo ratings
    if os.path.exists(get_elo_rating_dir(model_elo_rating_path)):
        elo_ratings = np.load(get_elo_rating_dir(model_elo_rating_path))
        models_elo_rating = elo_ratings["models"].tolist()
        best_models_elo_rating = elo_ratings["best_models"].tolist()
    else:
        models_elo_rating = [100]
        best_models_elo_rating = [100]
    # load competing model
    if os.path.exists(get_last_model_dir(model_path)):
        compete_model = load_model(get_last_model_dir(model_path))
    else:
        raise FileNotFoundError(f"Cannot find compete model at {get_last_model_dir(model_path)}")
    # load current best model
    if os.path.exists(get_last_model_dir(best_model_path)):
        best_model = load_model(get_last_model_dir(best_model_path))
    else:
        raise FileNotFoundError(f"Cannot find current best model at {get_last_model_dir(best_model_path)}")

    best_model_wins = 0

    for i in range(COMPARATOR_NO_OF_GAMES):
        start = datetime.now()
        best_model_board = Board(train_random=False)
        compete_model_board = Board(train_random=False)

        # play games
        while not best_model_board.is_terminal():
            assert np.array_equal(best_model_board.board, compete_model_board.board)
            if best_model_board.turn == (-1) ** i:
                policy = best_model_board.mcts(best_model, iter=COMPARATOR_MCTS_ITERS, cpuct=COMPARATOR_C_PUCT)
                move = best_model_board.get_mcts_move(policy)
                best_model_board = best_model_board.traverse(move[0], move[1])
                compete_model_board = compete_model_board.traverse(move[0], move[1])
            else:
                policy = compete_model_board.mcts(compete_model, iter=COMPARATOR_MCTS_ITERS, cpuct=COMPARATOR_C_PUCT)
                move = compete_model_board.get_mcts_move(policy)
                compete_model_board = compete_model_board.traverse(move[0], move[1])
                best_model_board = best_model_board.traverse(move[0], move[1])

        winner = best_model_board.winner()
        if winner == (-1) ** i:
            best_model_wins += 1
            winner_model = "Best model"
        else:
            winner_model = "Compete model"

        time_taken = datetime.now() - start
        print(name, f"Winner: {winner_model}, Game no: {i+1}, Run time: {time_taken}")

    best_model_win_rate = float(best_model_wins / COMPARATOR_NO_OF_GAMES)

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
