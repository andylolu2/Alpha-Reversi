from queue import Queue
import numpy as np
import random
import copy
import multiprocessing as mp
from constants import BOARD_DIM, C_PUCT, C_VIRTUAL_LOSS, HISTORY_LEN, MCTS_BATCH_SIZE, SAMPLES_PER_POS
from helper_methods import dihedral_trans

class Board:
    '''
        Class Board implements game rules and game tree traversal
    '''
    def __init__(self, board=None, turn=None, train_random=None, p=None, get_edges=False, history=None, move_num=0):
        self.visited = False
        self.move_num = move_num
        self.N = 0
        self.W = 0
        self.Q = 0
        self.P = p
        self.VIRTUAL_LOSS = 1
        if board is None:
            self.board = []
            for i in range(BOARD_DIM[0]):
                self.board.append([])
                for _ in range(BOARD_DIM[1]):
                    self.board[i].append([0])
            self.board = np.array(self.board, dtype=np.float32)
            self.reset()
        else:
            self.board = np.copy(board)
        
        if history is None:
            self.history = [np.zeros((8, 8, 1), dtype=np.float32) for _ in range(HISTORY_LEN-1)]
            self.history.append(np.copy(self.board))
        else:
            self.history = history[:]
        
        if turn is None:
            self.turn = 1
        else:
            self.turn = turn
        
        if train_random is None:
            self.train_random = True
        else:
            self.train_random = train_random

        self.parent = None

        self.edges = None
        if get_edges:
            self.edges = self.get_edges()

    def __copy__(self):
        return Board(board=self.board, turn=self.turn,
                            train_random=self.train_random, move_num=self.move_num,
                            history=self.history)
    
    def __repr__(self) -> str:
        return repr(self.board.reshape(BOARD_DIM))

    def _add(self, color, x, y):
        if x < 0 or x >= BOARD_DIM[0] or y < 0 or y >= BOARD_DIM[1]:
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

    def reset(self):
        for i in range(BOARD_DIM[0]):
            for j in range(BOARD_DIM[1]):
                self.board[i][j] = [0]
        self.board[int(BOARD_DIM[0] / 2)][int(BOARD_DIM[1] / 2)] = [1]
        self.board[int(BOARD_DIM[0] / 2) - 1][int(BOARD_DIM[1] / 2)] = [-1]
        self.board[int(BOARD_DIM[0] / 2)][int(BOARD_DIM[1] / 2) - 1] = [-1]
        self.board[int(BOARD_DIM[0] / 2) - 1][int(BOARD_DIM[1] / 2) - 1] = [1]      

    def traverse(self, x, y):
        '''
            Traverses the game tree
        '''
        self.edges = self.get_edges()
        if (x, y) in self.edges:
            child = self.edges[(x, y)]
            child.parent = None
            return child
        else:
            raise KeyError('The move is not valid')


    def get_edges(self):
        if self.edges is not None:
            return self.edges
        children = {}
        possible_moves = self.possible_moves(self.turn)
        for move in possible_moves:
            child = copy.copy(self)
            child._add(child.turn, move[0], move[1])
            child.parent = self
            children[move] = child
        return children
    
    def get_temperature(self):
        if self.train_random and self.move_num <= 10:
            return 1
        return 1e-1

    def possible(self, color, x, y):
        if x < 0 or x >= BOARD_DIM[0] or y < 0 or y >= BOARD_DIM[1]:
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
        for i in range(BOARD_DIM[0]):
            for j in range(BOARD_DIM[1]):
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
        if next_x < 0 or next_x >= BOARD_DIM[0] or next_y < 0 or next_y >= BOARD_DIM[1]:
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
        '''
            Output: an tuple of a {board_dim_0 * board_dim_1 * (HISTORY_LEN * 2 + 1) array} and an int representing the transformation
                (HISTORY_LEN * 2 + 1) comes from HISTORY_LEN layers for blacks, 
                HISTORY_LEN layers for whites and 1 layer to indicate player color
        '''
        # Separate black and white pieces
        black_history = []
        white_history = []
        for history in self.history:
            black_history.append(np.where(history == 1, history, 0))
            white_history.append(np.where(history == -1, history, 0))
        nn_input = np.concatenate((black_history, white_history), axis=2) * self.turn
        nn_input = nn_input.reshape(BOARD_DIM + (HISTORY_LEN * 2,))

        # Layer to indicate color
        color = 1 if self.turn == 1 else 0
        color = np.full(BOARD_DIM + (1,), color, dtype=np.float32)

        choice, transforamtion = dihedral_trans(nn_input)
        return (np.concatenate((choice, color), axis=2), transforamtion)

    def evaluate(self, nn):
        nn_inputs, transformations = tuple(zip(*[self.as_nn_input()]))
        nn_inputs = np.array(nn_inputs)
        outputs = nn(nn_inputs)
        values = []
        for _policy, _value, transformation in zip(outputs[0].numpy(), outputs[1].numpy(), transformations):
            _real_policy = dihedral_trans(_policy.reshape(BOARD_DIM), transformation=transformation, inverse=True)
            values.append(_value)
        value = sum(values) / len(values)
        return value

    def alpha_beta_value(self, nn, depth, a=-1e3, b=1e3, epsilon=0.05):
        if self.train_random and random.random() <= epsilon:
            return (None, random.choice(self.possible_moves(self.turn)))
        else:
            if depth == 0 or self.is_terminal():
                value = self.evaluate(nn)
                return (value, None)
            elif self.turn == 1:
                self.edges = self.get_edges()
                value = -1e9
                best_move = next(iter(self.edges.keys()))
                for move, child in self.edges.items():
                    children_value, _ = child.alpha_beta_value(nn, depth - 1, a=a, b=b, epsilon=0)
                    if children_value > value:
                        value = children_value
                        best_move = move
                    a = max(a, value)
                    if a >= b:
                        break
                return (value, best_move)
            elif self.turn == -1:
                self.edges = self.get_edges()
                value = 1e9
                best_move = next(iter(self.edges.keys()))
                for move, child in self.edges.items():
                    children_value, _ = child.alpha_beta_value(nn, depth - 1, a=a, b=b, epsilon=0)
                    if children_value < value:
                        value = children_value
                        best_move = move
                    b = min(b, value)
                    if a >= b:
                        break
                return (value, best_move)

    def mcts(self, nn, iter=24):
        '''
            Monte Carlo Tree Search algorithm
            Args:
                nn: the neural network used to evaluate the board state and returns a policy and value
                iter: number of monte carlo searches
            
            Returns: policy (BOARD_DIM[0] * BOARD_DIM[1] dimentional vector)
        '''
        if self.is_terminal():
            return np.zeros(BOARD_DIM, dtype=np.float32)
        
        set_nodes = set()
        for i in range(iter):
            # Select
            node = self
            while True:
                if node.edges is None:  # first visit
                    break
                elif node.edges == {}:  # terminal node
                    break
                else:
                    best_score = -1e3
                    best_move = None
                    for move, edge in node.edges.items():
                        U = C_PUCT * edge.P * ((node.N - 1) ** 0.5) / (1 + edge.N)
                        score = (edge.Q + U) * edge.VIRTUAL_LOSS
                        if score > best_score:
                            best_score = score
                            best_move = move
                    node = node.edges[best_move]
                    node.VIRTUAL_LOSS *= C_VIRTUAL_LOSS
            set_nodes.add(node)
            
            if i % MCTS_BATCH_SIZE == 0 or i == iter-1:
                nodes = list(set_nodes)
                
                # Evaluate
                nn_inputs, transformations = tuple(zip(*[node.as_nn_input() for node in nodes]))
                nn_inputs = np.array(nn_inputs)
                outputs = nn(nn_inputs)
                for node, _policy, _value, transformation in zip(nodes, outputs[0].numpy(), outputs[1].numpy(), transformations):
                    _real_policy, _ = dihedral_trans(_policy.reshape(BOARD_DIM), transformation=transformation, inverse=True)
                    
                    # Expand
                    node.edges = node.get_edges()
                    for move, edge in node.edges.items():
                        # N, W, Q values are implicitly 0
                        edge.P = _real_policy[move]
                    
                    # Backup
                    current_node = node
                    while True:
                        if current_node.parent is None:
                            break
                        current_node.N += 1
                        current_node.W += _value
                        current_node.Q = current_node.W / current_node.N
                        current_node.VIRTUAL_LOSS = 1
                        current_node = current_node.parent
                
                nodes = []
        
        # Calculate policy based on visit counts
        policy = np.zeros(BOARD_DIM, dtype=np.float32)
        temperature = self.get_temperature()
        visit_sum = 0
        for move, edge in self.edges.items():
            visit_score = edge.N ** (1 / temperature)
            policy[move] = visit_score
            visit_sum += visit_score
        policy /= visit_sum
        policy = policy.reshape(BOARD_DIM[0] * BOARD_DIM[1])
        return policy
        
    def get_mcts_move(self, policy):
        '''
            policy: 2D np array
            returns: move, tuple of (x, y)
        '''
        move = np.random.choice(np.arange(BOARD_DIM[0] * BOARD_DIM[1]), p=policy)
        x = int(move / BOARD_DIM[0])
        y = move % BOARD_DIM[0]
        return (x, y)