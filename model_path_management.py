import os

MODEL_NAME = "Trial0"
model_path = "models\\{}\\{}_{}.tf"
best_model_path = "best_models\\{}\\{}_{}.tf"
model_elo_rating_path = "model_elo_rating\\{}\\{}.npz"
training_data_dir = "training_data\\{}".format(MODEL_NAME)


def get_last_model_index(path, name=MODEL_NAME):
    index = 0
    while os.path.exists(path.format(name, name, index)):
        index += 1
    index -= 1
    return index


def get_next_model_dir(path, name=MODEL_NAME):
    return path.format(name, name, get_last_model_index(path, name) + 1)


def get_last_model_dir(path, name=MODEL_NAME):
    return path.format(name, name, get_last_model_index(path, name))


def get_elo_rating_dir(path, name=MODEL_NAME):
    return path.format(name, name)