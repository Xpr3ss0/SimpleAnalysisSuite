import numpy as np


def filter_saturated(img, bits=12):
    """Creates a mask for pixels that have the max value."""
    max_val = 2**bits - 1
    return img == max_val