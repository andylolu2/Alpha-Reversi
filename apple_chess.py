import numpy as np
import random
import multiprocessing as mp

C_PUCT = 1.5


class Board:
    def __init__(self, board=None, turn=None, train_random=None, p=None, get_edges=False, history=None, move_num = 0):
        self.board_dim = (8, 8)
        self.visited = False
        self.move_num = move_num
        self.N = 0
        self.W = 0
        self.Q = 0
        self.P = p
        if board is None:
            self.board = []
            for i in range(self.board_dim[0]):
                self.board.append([])
                for j in range(self.board_dim[1]):
                    self.board[i].append([0])
            self.board = np.array(self.board, dtype=np.float32)
            self.reset()
        else:
            self.board = board
        if history is None:
            self.history = [np.zeros((8, 8, 1), dtype=np.float32) for _ in range(3)]
            self.history.append(np.copy(self.board))
        else:
            self.history = history
        if turn is None:
            self.turn = 1
        else:
            self.turn = turn
        if train_random is None:
            self.train_random = True
        else:
            self.train_random = train_random

        self.edges = []
        if get_edges:
            self.edges = self.get_edges()

    def reset(self):
        for row in self.board:
            for element in row:
                element = [0]
        self.board[int(self.board_dim[0] / 2)][int(self.board_dim[1] / 2)] = [1]
        self.board[int(self.board_dim[0] / 2) - 1][int(self.board_dim[1] / 2)] = [-1]
        self.board[int(self.board_dim[0] / 2)][int(self.board_dim[1] / 2) - 1] = [-1]
        self.board[int(self.board_dim[0] / 2) - 1][int(self.board_dim[1] / 2) - 1] = [1]

    def add(self, color, x, y):
        if x < 0 or x >= self.board_dim[0] or y < 0 or y >= self.board_dim[1]:
            print("That is outside of the grid")
        elif self.board[x][y][0] == 0:
            if self.possible(color, x, y):
                self.board[x][y][0] = color
                self.flip_and_find_all_direction(color, x, y)
                self.turn *= -1
                self.move_num += 1
                self.history.append(np.copy(self.board))
                self.history.pop(0)
                if len(self.possible_moves(self.turn)) == 0:
                    self.turn *= -1
            else:
                print("Move is invalid")
        else:
            print("Position already occupied")

    def get_edges(self):
        children = []
        possible_moves = self.possible_moves(self.turn)
        for move in possible_moves:
            child = Board(board=np.copy(self.board), turn=self.turn,
                          train_random=self.train_random, move_num=self.move_num,
                          history=self.history)
            child.add(child.turn, move[0], move[1])
            children.append((child, move))
        return children

    def possible(self, color, x, y):
        if x < 0 or x >= self.board_dim[0] or y < 0 or y >= self.board_dim[1]:
            return False
        elif self.board[x][y][0] == 0:
            self.board[x][y][0] = color
            temp = np.copy(self.board)
            self.flip_and_find_all_direction(color, x, y)
            if np.all(temp == self.board):
                self.board = temp
                self.board[x][y][0] = 0
                return False
            else:
                self.board = temp
                self.board[x][y][0] = 0
                return True
        else:  # already occupied
            return False

    def possible_moves(self, color):
        possible_moves = []
        for i in range(self.board_dim[0]):
            for j in range(self.board_dim[1]):
                if self.board[i][j] != 0:
                    for x_dir in range(-1, 2):
                        for y_dir in range(-1, 2):
                            if not (x_dir == 0 and y_dir == 0):
                                if self.possible(color, i + x_dir, j + y_dir):
                                    if (i + x_dir, j + y_dir) not in possible_moves:
                                        possible_moves.append((i + x_dir, j + y_dir))
        return possible_moves

    def flip_and_find_direction(self, color, x, y, direction):
        next_x = x + direction[0]
        next_y = y + direction[1]
        if next_x < 0 or next_x >= self.board_dim[0] or next_y < 0 or next_y >= self.board_dim[1]:
            return False
        else:
            next_item = self.board[next_x][next_y][0]
            if next_item == 0:
                return False
            elif next_item == color:
                return True
            elif next_item == -color:
                next_exist = self.flip_and_find_direction(color, next_x, next_y, direction)
                if next_exist:
                    self.board[next_x][next_y][0] = color
                return next_exist
            else:
                print("Error in find_next_dir")

    def flip_and_find_all_direction(self, color, x, y):
        for x_dir in range(-1, 2):
            for y_dir in range(-1, 2):
                if not (x_dir == 0 and y_dir == 0):
                    self.flip_and_find_direction(color, x, y, (x_dir, y_dir))

    def log(self):
        temp = []
        for i in range(self.board_dim[0]):
            temp.append([])
            for j in range(self.board_dim[1]):
                temp[i].append(self.board[i][j][0])
        temp = np.array(temp)
        print(temp)

    def is_terminal(self):
        if len(self.possible_moves(1)) == 0 and len(self.possible_moves(-1)) == 0:
            return True
        else:
            return False

    def winner(self):
        if self.is_terminal():
            whites, blacks = self.black_and_white_num()
            if blacks > whites:
                return -1
            elif whites > blacks:
                return 1
            elif whites == blacks:
                return 0
        else:
            print("Error: Game hasn't completed")

    def black_and_white_num(self):
        whites = 0
        blacks = 0
        for row in self.board:
            for col in row:
                if col[0] == 1:
                    whites += 1
                if col[0] == -1:
                    blacks += 1
        return whites, blacks

    def as_nn_input(self):
        nn_input = np.copy(self.board) * self.turn
        random_indices = random.sample(range(8), 4)
        if self.turn == 1:
            color = 1
        else:
            color = 0
        color = np.full_like(nn_input, color, dtype=np.float32)
        nn_inputs = []
        for i in random_indices:
            if i == 0:
                choice = nn_input
                nn_inputs.append(np.concatenate((choice, color), axis=2))
            elif i == 1:
                choice = np.rot90(nn_input, 1)
                nn_inputs.append(np.concatenate((choice, color), axis=2))
            elif i == 2:
                choice = np.rot90(nn_input, 2)
                nn_inputs.append(np.concatenate((choice, color), axis=2))
            elif i == 3:
                choice = np.rot90(nn_input, -1)
                nn_inputs.append(np.concatenate((choice, color), axis=2))
            elif i == 4:
                choice = np.fliplr(nn_input)
                nn_inputs.append(np.concatenate((choice, color), axis=2))
            elif i == 5:
                choice = np.flipud(nn_input)
                nn_inputs.append(np.concatenate((choice, color), axis=2))
            elif i == 6:
                choice = np.rot90(np.fliplr(nn_input), 1)
                nn_inputs.append(np.concatenate((choice, color), axis=2))
            elif i == 7:
                choice = np.rot90(np.fliplr(nn_input), -1)
                nn_inputs.append(np.concatenate((choice, color), axis=2))
        return nn_inputs

    def alpha_beta_value(self, func, nn, depth, a, b, max_depth=-1, epsilon=0.05):
        try:
            if self.train_random and random.random() < epsilon and depth == max_depth:
                return [None, random.choice(self.possible_moves(self.turn))]
            else:
                if depth == 0 or self.is_terminal():
                    value = func(self, nn)
                    return [value, None]
                elif self.turn == 1:
                    self.edges = self.get_edges()
                    value = -1e9
                    best_move = self.edges[0][1]
                    for child, move in self.edges:
                        children_value = child.alpha_beta_value(func, nn, depth - 1, a, b)[0]
                        if children_value > value:
                            value = children_value
                            best_move = move
                        a = max(a, value)
                        if a >= b:
                            break
                    return [value, best_move]
                elif self.turn == -1:
                    self.edges = self.get_edges()
                    value = 1e9
                    best_move = self.edges[0][1]
                    for child, move in self.edges:
                        children_value = child.alpha_beta_value(func, nn, depth - 1, a, b)[0]
                        if children_value < value:
                            value = children_value
                            best_move = move
                        b = min(b, value)
                        if a >= b:
                            break
                    return [value, best_move]
        except Exception as e:
            print(e, flush=True)
