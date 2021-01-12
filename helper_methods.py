from random import randint
from model_path_management import training_data_dir_0, training_data_dir_1
import numpy as np
import time
from tensorflow import keras


def dihedral_trans(np_array, transformation=None, inverse=False):
    if isinstance(transformation, int) and 0 <= transformation < 8:
        i = transformation
        if inverse:
            i *= -1
    else:
        i = randint(0, 7)
    if i == 0:
        return (np_array, i)
    elif i == 1 or i == -3:
        return (np.rot90(np_array, 1), i)
    elif i == 2 or i == -2:
        return (np.rot90(np_array, 2), i)
    elif i == 3 or i == -1:
        return (np.rot90(np_array, -1), i)
    elif i == 4 or i == -4:
        return (np.fliplr(np_array), i)
    elif i == 5 or i == -5:
        return (np.flipud(np_array), i)
    elif i == 6 or i == -6:
        return (np.rot90(np.fliplr(np_array), 1), i)
    elif i == 7 or i == -7:
        return (np.rot90(np.fliplr(np_array), -1), i)

def load_model(dir):
    try:
        print(f"{dir} loaded!")
        return keras.models.load_model(dir, compile=False)
    except Exception:
        try:
            time.sleep(3)
            print(f"{dir} loaded!")
            return keras.models.load_model(dir, compile=False)
        except Exception:
            raise FileNotFoundError(f"Failed find model at {dir}")

def load_training_data():
    try:
        training_data = np.load(training_data_dir_0)
        return (training_data["data_state"], training_data["data_policy"], training_data["data_value"])
    except Exception:
        training_data = np.load(training_data_dir_1)
        return (training_data["data_state"], training_data["data_policy"], training_data["data_value"])
