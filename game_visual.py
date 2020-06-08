from mttkinter.mtTkinter import *
import tensorflow.keras as K
from apple_chess import Board
from model_path_management import *
from multiprocessing.pool import ThreadPool
import numpy as np
import math


def value_func(board, nn):
    value = np.mean(nn(np.array(board.as_nn_input()), training=False).numpy())
    return value


def draw(canvas, board):
    grid_unit = dimension / 8
    global margin
    padding = 10
    for i in range(8):
        for j in range(8):
            canvas.create_rectangle(i * grid_unit + margin, j * grid_unit + margin,
                                    (i + 1) * grid_unit + margin, (j + 1) * grid_unit + margin,
                                    fill="#f2ca5c",
                                    outline="black", width=1)
    for i in range(8):
        for j in range(8):
            board_pixel = board[i][j][0]
            if board_pixel == 1:
                canvas.create_oval(i * grid_unit + padding + margin, j * grid_unit + padding + margin,
                                   (i + 1) * grid_unit - padding + margin, (j + 1) * grid_unit - padding + margin,
                                   fill="white", outline="black", width=2)
            elif board_pixel == -1:
                canvas.create_oval(i * grid_unit + padding + margin, j * grid_unit + padding + margin,
                                   (i + 1) * grid_unit - padding + margin, (j + 1) * grid_unit - padding + margin,
                                   fill="black", outline="black", width=2)


def mouse_clicked(event):
    if Test.turn == 1:
        grid_unit = dimension / 8
        possible_moves = Test.possible_moves(Test.turn)
        target = (math.floor(event.x / grid_unit), math.floor(event.y / grid_unit))
        if target in possible_moves:
            Test.add(Test.turn, target[0], target[1])
            draw(tk, Test.board)
            tk.update()
            player_moved.set(True)


def key_pressed(event):
    if Test.is_terminal():
        finish.set(True)


def draw_moves(canvas, moves):
    global margin
    grid_unit = dimension / 8
    for m in moves:
        canvas.create_rectangle(m[0] * grid_unit + margin, m[1] * grid_unit + margin,
                                (m[0] + 1) * grid_unit + margin, (m[1] + 1) * grid_unit + margin,
                                fill="#81f25c", outline="black", width=1)


def show_end_game_window(canvas):
    global dimension
    global Test
    winner = Test.winner()
    winner_name = None
    if winner == 1:
        winner_name = "White"
    elif winner == -1:
        winner_name = "Black"
    white_num, black_num = Test.black_and_white_num()
    width = 230
    height = 90
    center = dimension / 2
    vertical_offset = 20
    canvas.create_rectangle(center - width / 2, center - height / 2 - vertical_offset,
                            center + width / 2, center + height / 2 - vertical_offset,
                            fill="#2d3445", outline="#2d3445")
    canvas.create_text(center, center - vertical_offset,
                       text="{} Won! \n White {} : {} Black"
                            " \n Press Enter to exit...".format(winner_name, white_num, black_num),
                       font=("Arial", 15), fill="#fcfdff", justify=CENTER)
    canvas.update()


def apply_AI_result(a_b_result):
    print("Move: {:2d}, Win rate of AI: {:.2%}".format(Test.move_num, (a_b_result[0] / 2) * (-1) + 0.5))
    move = a_b_result[1]
    Test.add(-1, move[0], move[1])
    draw(tk, Test.board)
    tk.update()


def finish_thinking(event):
    AI_thinking.set(False)


def AI_evaluate(B, func, nn, depth, a, b):
    try:
        res = B.alpha_beta_value(func, nn, depth, a, b)
        return res
    except Exception as e:
        print(e, flush=True)


if __name__ == "__main__":
    if os.path.exists(get_last_model_dir(best_model_path)):
        model = K.models.load_model(get_last_model_dir(best_model_path))
        print("best model {} loaded!".format(get_last_model_dir(best_model_path)))
    else:
        print("model not found")

    Test = Board(train_random=False)
    master = Tk()
    dimension = 800
    margin = 10
    player_moved = BooleanVar()
    AI_thinking = BooleanVar()
    finish = BooleanVar()
    finish.set(False)
    tk = Canvas(master, width=dimension + margin * 2, height=dimension + margin * 2)
    tk.bind("<Button-1>", mouse_clicked)
    tk.bind("<Return>", key_pressed)
    tk.pack()

    pool = ThreadPool(processes=1)

    PLAYER_IS_WHITE = True
    if PLAYER_IS_WHITE:
        C_GO_FIRST = 1
    else:
        C_GO_FIRST = -1

    draw(tk, Test.board)
    tk.update()

    while True:
        if Test.is_terminal():
            tk.focus_set()
            print("{} won!".format(Test.winner()))
            show_end_game_window(tk)
            tk.wait_variable(finish)
            break
        else:
            if Test.turn * C_GO_FIRST == 1:
                player_moved.set(False)
                draw_moves(tk, Test.possible_moves(Test.turn))
                # listen to mouse click
                tk.wait_variable(player_moved)
            elif Test.turn * C_GO_FIRST == -1:
                a_move = pool.apply_async(func=AI_evaluate, args=(Test, value_func, model, 2, -1e3, 1e3),
                                          callback=finish_thinking)
                tk.wait_variable(AI_thinking)
                apply_AI_result(a_move.get())
