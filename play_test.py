import os
import tensorflow as tf
import tensorflow.keras as K
from apple_chess import Board
import numpy as np

MODEL_NAME = "Trial6"
model_dir = "models\\{}.tf".format(MODEL_NAME)

if os.path.exists(model_dir):
    model = K.models.load_model(model_dir)
    print("model loaded!")
else:
    print("model not found")


def value_func(board, nn):
    value = np.mean(nn(np.array(board.as_nn_input()), training=False).numpy())
    return value


Test = Board(train_random=False)


while True:
    print("{}'s turn".format(Test.turn))
    AI_result = Test.alpha_beta_value(value_func, model, 4, -1e3, 1e3)
    print("Win rate of AI: {:.2%}".format((AI_result[0]/2)*(-1) + 0.5))
    Test.log()
    if Test.is_terminal():
        print("{} won!".format(Test.winner()))
        break
    else:
        if Test.turn == 1:
            print("Possible moves: {}".format(Test.possible_moves(Test.turn)))
            valid = False
            while not valid:
                try:
                    text = input("Place: ")
                    x_ = int(text.split(" ")[0])
                    y_ = int(text.split(" ")[1])
                    Test.add(Test.turn, x_, y_)
                    valid = True
                except:
                    valid = False
                    print("Input not valid. Should be in format of 'y x'")
        elif Test.turn == -1:
            move = AI_result[1]
            Test.add(-1, move[0], move[1])
            print("AI placed {} {}".format(move[0], move[1]))
