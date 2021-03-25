### Constants in Board class

# Board dimension
BOARD_DIM = (8, 8)
# Number of board positions to use as NN input
HISTORY_LEN = 2
# Number of samples taken per board position 
SAMPLES_PER_POS = 1
# Determines the degree of exploration in MCTS
C_PUCT = 0.3
# MCTS batch size
MCTS_BATCH_SIZE = 4
# MCTS virtual loss factor
C_VIRTUAL_LOSS = 0.8


### Constants used in training

# Dataset sample size
TRAINING_SAMPLE_SIZE = 4096
# Minimum dataset size
MIN_DATASET_SIZE = TRAINING_SAMPLE_SIZE * 2
# Total number of train steps
NO_OF_TRAININGS = 100_000
# Number of train steps between each save of the training model
SAVE_MODEL_EVERY = 250
# Number of train steps between making the training model compete with current best
COMPETE_MODEL_EVERY = 500
# Number of train steps between each load of new training data
LOAD_DATA_EVERY = 100
# Number of epochs between selecting new slice of training data
EPOCHS = 1
# The weighting for value loss
VALUE_LOSS_WEIGHTING = 2
# Learning rate of optimizer
LEARNING_RATE = 1e-4
# Momentum of optimizer
MOMENTUM = 0.9
# Batch size
BATCH_SIZE = 64
# Coefficient for L2 loss to prevent overfitting
C_L2 = 1e-4


### Constants used in self-play

# Self-play mcts iterations
SELF_PLAY_MCTS_ITERS = 4
# Self-play exploration level
SELF_PLAY_C_PUCT = 0.3
# Number of games to self-play
NO_OF_GAMES = 10_000
# The threshold to end clearly-lost games
V_RESIGN = 0.1
# Number of games between each save of new data
SAVE_DATA_EVERY = 20
# Number of games between loading better model
UPDATE_MODEL_EVERY = 20
# Maximum number of data points to save
MAX_TRAINING_SIZE = 140_000


### Constants for model architecture

# Number of residual blocks
NO_OF_RES_BLOCKS = 4
# Number of filters per conv layer
NO_OF_FILTERS = 256


### Constants for comparator

# Constant for scaling model elo
C_ELO = 1/400
# Number of games to play to compare models
COMPARATOR_NO_OF_GAMES = 31
# Comparator MCTS iterations
COMPARATOR_MCTS_ITERS = 12
# Comparotor MCTS exploration level
COMPARATOR_C_PUCT = 0.3
# The winrate require for new model to take over current best
WINRATE_CUTOFF = 0.55


### Game UI constants

# Dimension of the GUI window
DIMENSION = 720
# Margin of Piece to edge of square in board grid
MARGIN = 10
