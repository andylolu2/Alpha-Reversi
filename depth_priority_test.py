from helper_methods import load_model
from apple_chess import Board
from model_path_management import best_model_path, get_last_model_dir
import tensorflow as tf
import numpy as np
import os
from datetime import datetime
from multiprocessing import Pool

def play_games(model_path, number_of_games, starting_game_index):
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)

    model = load_model(model_path)
    winners = []
    for i in range(number_of_games):
        start = datetime.now()
        ord_model_board = Board(train_random=False)
        depth_prior_board = Board(train_random=False)

        # play games
        while not ord_model_board.is_terminal():
            assert np.array_equal(ord_model_board.board, depth_prior_board.board)
            if ord_model_board.turn == (-1) ** i:
                policy = ord_model_board.mcts(model, iter=64, cpuct=0.2)
                move = ord_model_board.get_mcts_move(policy)
                ord_model_board = ord_model_board.traverse(move[0], move[1])
                depth_prior_board = depth_prior_board.traverse(move[0], move[1])
            else:
                policy = depth_prior_board.mcts(model, iter=64, cpuct=0.2, depth_weight=1.3)
                move = depth_prior_board.get_mcts_move(policy)
                depth_prior_board = depth_prior_board.traverse(move[0], move[1])
                ord_model_board = ord_model_board.traverse(move[0], move[1])
                

        winner = ord_model_board.winner()
        str_winner = "ordinary" if winner == (-1) ** i else "depth_prior"
        winners.append(str_winner)

        print(f"Winner: {str_winner}, Game length: {ord_model_board.move_num}, Game no: {starting_game_index + i}, Run time: {datetime.now() - start}")

    return winners

NO_OF_GAMES_ = 300
SAVE_DATA_EVERY_ = 25

def self_play(workers=1):
    self_play_res = []
    winners = []
    with Pool(processes=workers, maxtasksperchild=2) as pool:
        for i in range(1, NO_OF_GAMES_ + 1):
            if i % SAVE_DATA_EVERY_ == 1:
                no_of_games = int(SAVE_DATA_EVERY_/workers)
                for j in range(workers - 1):
                    self_play_res.append(pool.apply_async(play_games, args=(get_last_model_dir(best_model_path), no_of_games, i + j * no_of_games)))
                self_play_res.append(pool.apply_async(play_games, args=(
                    get_last_model_dir(best_model_path), 
                    no_of_games + SAVE_DATA_EVERY_ % workers, 
                    i + (workers - 1) * no_of_games)))
                
                # Collect pool results
                for res in self_play_res:
                    winners += res.get()
                self_play_res.clear()

                ordinary_win = 0
                depth_prior_win = 0
                for winner in winners:
                    if winner == "ordinary": ordinary_win += 1
                    else: depth_prior_win += 1
                print(f"ordinary : depth prior = {ordinary_win} : {depth_prior_win}")
                print(f"Percentage: {100 * max(ordinary_win, depth_prior_win)/(ordinary_win + depth_prior_win)}%")


    print("completed!")
    return

if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    self_play(workers=4)