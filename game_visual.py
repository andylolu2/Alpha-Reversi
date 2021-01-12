from helper_methods import dihedral_trans
from constants import BOARD_DIM, C_PUCT, DIMENSION, MARGIN
import os
from model_path_management import get_last_model_dir, best_model_path
import mttkinter.mtTkinter as mtk
import tensorflow as tf
from apple_chess import Board
from multiprocessing.pool import ThreadPool
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import math

Game = Board(train_random=False, deploy=True)

def _from_rgb(rgb):
    """translates an rgb tuple of int to a tkinter friendly color code
    """
    r, g, b = rgb
    return f'#{r:02x}{g:02x}{b:02x}'

def draw_grid(canvas):
    grid_unit = DIMENSION / 8
    for i in range(8):
        for j in range(8):
            canvas.create_rectangle(i * grid_unit + MARGIN, j * grid_unit + MARGIN,
                                    (i + 1) * grid_unit + MARGIN, (j + 1) * grid_unit + MARGIN,
                                    fill="#f2ca5c",
                                    outline="black", width=1)

def draw_pieces(canvas, board):
    grid_unit = DIMENSION / 8
    padding = 10
    for i in range(8):
        for j in range(8):
            board_pixel = board[i][j][0]
            if board_pixel == 1:
                canvas.create_oval(i * grid_unit + padding + MARGIN, j * grid_unit + padding + MARGIN,
                                   (i + 1) * grid_unit - padding + MARGIN, (j + 1) * grid_unit - padding + MARGIN,
                                   fill="white", outline="black", width=3)
            elif board_pixel == -1:
                canvas.create_oval(i * grid_unit + padding + MARGIN, j * grid_unit + padding + MARGIN,
                                   (i + 1) * grid_unit - padding + MARGIN, (j + 1) * grid_unit - padding + MARGIN,
                                   fill="black", outline="black", width=3)

def draw_policy(canvas, policy):
    grid_unit = DIMENSION / 8
    for i in range(len(policy)):
        for j in range(len(policy[i])):
            if policy[i, j] > 0:
                canvas.create_rectangle(i * grid_unit + MARGIN, j * grid_unit + MARGIN,
                                        (i + 1) * grid_unit + MARGIN, (j + 1) * grid_unit + MARGIN,
                                        fill=_from_rgb((255, int(170 - (170-89) * policy[i, j]), int(125 - 125 * policy[i, j]))), 
                                        outline="black", width=1)

def draw_moves(canvas, moves):
    grid_unit = DIMENSION / 8
    for m in moves:
        canvas.create_rectangle(m[0] * grid_unit + MARGIN, m[1] * grid_unit + MARGIN,
                                (m[0] + 1) * grid_unit + MARGIN, (m[1] + 1) * grid_unit + MARGIN,
                                fill="#81f25c", outline="black", width=1, stipple="gray75")

def mouse_clicked(event):
    global Game
    if Game.turn * C_GO_FIRST == 1:
        grid_unit = DIMENSION / 8
        possible_moves = Game.possible_moves(Game.turn)
        target = (math.floor(event.x / grid_unit), math.floor(event.y / grid_unit))
        if target in possible_moves:
            Game = Game.traverse(target[0], target[1])
            draw_grid(tk)
            draw_pieces(tk, Game.board)
            tk.update()
            player_moved.set(True)


def key_pressed(event):
    global finish
    finish.set(True)


def show_end_game_window(canvas):
    global Game
    winner = Game.winner()
    winner_name = None
    if winner == 1:
        winner_name = "White"
    elif winner == -1:
        winner_name = "Black"
    white_num, black_num = Game.black_and_white_num()
    width = 230
    height = 90
    center = DIMENSION / 2
    vertical_offset = 20
    canvas.create_rectangle(center - width / 2, center - height / 2 - vertical_offset,
                            center + width / 2, center + height / 2 - vertical_offset,
                            fill="#2d3445", outline="#2d3445")
    canvas.create_text(center, center - vertical_offset,
                       text="{} Won! \n White {} : {} Black"
                            " \n Press Enter to exit...".format(winner_name, white_num, black_num),
                       font=("Arial", 15), fill="#fcfdff", justify=mtk.CENTER)
    canvas.update()


def apply_ai_result(policy):
    global Game
    policy_2d = policy.reshape(BOARD_DIM)
    move = Game.get_mcts_move(policy)
    Game = Game.traverse(move[0], move[1])
    draw_grid(tk)
    draw_policy(tk, policy_2d)
    draw_pieces(tk, Game.board)
    tk.update()
    return Game


def finish_thinking(event):
    AI_thinking.set(False)


if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)
    if os.path.exists(get_last_model_dir(best_model_path)):
        model = tf.keras.models.load_model(get_last_model_dir(best_model_path), compile=False)
        print("best model {} loaded!".format(get_last_model_dir(best_model_path)))
    else:
        raise FileNotFoundError("There is no model to load")

    master = mtk.Tk()
    player_moved = mtk.BooleanVar()
    AI_thinking = mtk.BooleanVar()
    finish = mtk.BooleanVar()
    finish.set(False)
    tk = mtk.Canvas(master, width=DIMENSION + MARGIN * 2, height=DIMENSION + MARGIN * 2)
    tk.pack()

    pool = ThreadPool(processes=1)

    PLAYER_IS_WHITE = True
    C_GO_FIRST = 1 if PLAYER_IS_WHITE else -1

    tk.bind("<Button-1>", lambda event: mouse_clicked(event))
    tk.bind("<Return>", lambda event: key_pressed(event))
    draw_grid(tk)
    draw_pieces(tk, Game.board)
    tk.update()

    while True:
        tk.focus_set()
        # if Game.edges:
        #     print([(move, edge.Q) for move, edge in Game.edges.items()])
        if finish.get():
            break
        if Game.is_terminal():
            tk.focus_set()
            print("{} won!".format(Game.winner()))
            show_end_game_window(tk)
            tk.wait_variable(finish)
            break
        else:
            if Game.turn * C_GO_FIRST == 1:
                player_moved.set(False)
                draw_moves(tk, Game.possible_moves(Game.turn))
                # listen to mouse click
                tk.wait_variable(player_moved)
            elif Game.turn * C_GO_FIRST == -1:
                policy = pool.apply_async(func=Game.mcts, args=(model,),
                                            kwds={"iter": 128, "cpuct": C_PUCT, "verbose": True, "depth_weight": 1.3},
                                        callback=finish_thinking)
                tk.wait_variable(AI_thinking)
                policy = policy.get()
                # print(policy.round(decimals=3).reshape(BOARD_DIM).T)
                Game = apply_ai_result(policy)
