### Constants in Board class

# Board dimension
BOARD_DIM = (8, 8)
# Number of board positions to use as NN input
HISTORY_LEN = 4
# Number of samples taken per board position 
SAMPLES_PER_POS = 1
# Determines the degree of exploration in MCTS
C_PUCT = 1.5
# MCTS batch size
MCTS_BATCH_SIZE = 8
# MCTS virtual loss factor
C_VIRTUAL_LOSS = 0.7


### Constants used in training

# Dataset sample size
TRAINING_SAMPLE_SIZE = 2048
# Minimum dataset size
MIN_DATASET_SIZE = TRAINING_SAMPLE_SIZE * 2
# Total number of train steps
NO_OF_TRAININGS = 10_000
# Number of train steps between each save of the training model
SAVE_MODEL_EVERY = 20
# Number of train steps between making the training model compete with current best
COMPETE_MODEL_EVERY=50
# Number of train steps between each load of new training data
LOAD_DATA_EVERY = 25
# Number of epochs between selecting new slice of training data
EPOCHS = 1
# Learning rate of optimizer
LEARNING_RATE = 1e-4
# Momentum of optimizer
MOMENTUM = 0.9
# Batch size
BATCH_SIZE = 32
# Coefficient for L2 loss to prevent overfitting
C_L2 = 1e-4


### Constants used in self-play

# Number of games to self-play
NO_OF_GAMES = 10_000
# The threshold to end clearly-lost games
V_RESIGN = 0.1
# Number of games between each save of new data
SAVE_DATA_EVERY = 50
# Number of games between loading better model
UPDATE_MODEL_EVERY = 50
# Maximum number of data points to save
MAX_TRAINING_SIZE = 50_000


### Constants for model architecture

# Number of residual blocks
NO_OF_RES_BLOCKS = 1

# Constant for scaling model elo
C_ELO = 1/400

# Number of games to play to compare models
NO_OF_GAMES = 20

# The winrate require for new model to take over current best
WINRATE_CUTOFF = 0.6