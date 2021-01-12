from model import create_model
from constants import BOARD_DIM, MAX_TRAINING_SIZE, NO_OF_GAMES, SAVE_DATA_EVERY, SELF_PLAY_C_PUCT, SELF_PLAY_MCTS_ITERS, UPDATE_MODEL_EVERY
from helper_methods import dihedral_trans, load_model, load_training_data
from apple_chess import Board
from model_path_management import best_model_path, get_last_model_dir, model_path, get_next_model_dir, training_data_dir_0, training_data_dir
import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import pathlib
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool, Queue

def play_games(model_path, number_of_games, starting_game_index):
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)

    model = load_model(model_path)
    states = policies = values = []
    for i in range(number_of_games):
        temp_states = temp_policies = temp_values = temp_turns = []
        start = datetime.now()
        B = Board(train_random=True)

        # play games
        while not B.is_terminal():
            turn = B.turn
            state, transformation = B.as_nn_input()
            policy = B.mcts(model, iter=SELF_PLAY_MCTS_ITERS, cpuct=SELF_PLAY_C_PUCT)
            move = B.get_mcts_move(policy)
            B = B.traverse(move[0], move[1])

            # Save training data
            temp_states.append(state)
            policy, _ = dihedral_trans(policy.reshape(BOARD_DIM), transformation=transformation)
            policy = policy.reshape(BOARD_DIM[0] * BOARD_DIM[1])
            temp_policies.append(policy)
            temp_turns.append(turn)

        winner = B.winner()
        temp_values = [[winner * turn] for turn in temp_turns]

        states += temp_states
        policies += temp_policies
        values += temp_values

        assert len(states) == len(policies) == len(values), "The length of data does not match"
        print(f"Winner: {winner:+}, Game length: {len(temp_states)}, Game no: {starting_game_index + i}, Run time: {datetime.now() - start}")

    return (states, policies, values)

def self_play(workers=1):
    # create model
    if not os.path.exists(get_last_model_dir(best_model_path)):
        model = create_model()
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
    # load previous training data
    if os.path.exists(training_data_dir_0):
        data_state, data_policy, data_value = load_training_data()
        data_state = list(data_state)
        data_policy = list(data_policy)
        data_value = list(data_value)
    else:
        data_state = []
        data_policy = []
        data_value = []
        print("Couldn't find training data. Creating new")

    self_play_res = []
    with Pool(processes=workers) as pool:
        for i in range(1, NO_OF_GAMES + 1):
            if i % SAVE_DATA_EVERY == 1:
                no_of_games = int(SAVE_DATA_EVERY/workers)
                for j in range(workers - 1):
                    self_play_res.append(pool.apply_async(play_games, args=(get_last_model_dir(best_model_path), no_of_games, i + j * no_of_games)))
                self_play_res.append(pool.apply_async(play_games, args=(
                    get_last_model_dir(best_model_path), 
                    no_of_games + SAVE_DATA_EVERY % workers, 
                    i + (workers - 1) * no_of_games)))
                
                # Collect pool results
                for res in self_play_res:
                    temp_states, temp_policies, temp_values = res.get()
                    data_state += temp_states
                    data_policy += temp_policies
                    data_value += temp_values
                self_play_res.clear()

                assert len(data_state) == len(data_policy) == len(data_value), "The length of data does not match"

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

    print("completed!")
    return

if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    self_play(workers=1)