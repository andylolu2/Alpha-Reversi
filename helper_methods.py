from random import randint
import numpy as np


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
