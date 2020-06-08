# Apple_Chess_Zero
Implements part of Alpha Zero's algorithm for training good player in Reversi (Apple Chess)
-------------------------------------------------------------------------------------------
apple_chess.py:   Implements the Reversi (Apple_chess) game class and the alpha-beta pruning algorithm

self_play.py:     Loads the current best model (neural network) to play against itself to generate data (a pair of data = (the board state,                   the winner of the game)) for training

training.py:      Trains the current model using the data from self_play.py. The neural network takes the board state as input the outputs                   a prediction of the outcome of the game (would black or white win)

compare_models.py: After a certain amount of training by training.py, takes the newly trained model and make it fight against the current                     best model. If it wins with a winrate larger than 60%, it becomes the new best model for self_play.py

model_path_management.py: Manages the paths to save and load training data, models, and training results

visualise_model_elo.py: Shows a graph of the elo rating of the models against the model version

visualise_training_results.py: Shows a graph of the loss of the model aginst the number of training epochs

game_visual.py: Starts a tkinter UI to play against the current best model
