import numpy as np


def filter_saturated(img, bits=12):
    """Creates a mask for pixels that have the max value."""
    max_val = 2**bits - 1
    return img == max_val

def filter_above_threshold(img, threshold):
    """Creates a mask for pixels above a certain threshold."""
    return img > threshold