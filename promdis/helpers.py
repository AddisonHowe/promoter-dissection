"""General helper functions

"""

import numpy as np


def binary_arr_to_int(bin_arr):
    """Convert one or more binary arrays to integer value(s).
    """
    k = bin_arr.shape[-1]
    weights = 1 << np.arange(k)[::-1]
    return bin_arr @ weights


def int_to_binary_arr(int_array, n=None):
    """Convert one or more integers to binary arrays of length n.
    """
    if not isinstance(int_array, np.ndarray):
        int_array = np.array(int_array)
    if n is None:
        n = np.max([int_array.max(), 1]).item().bit_length()
    return (
        (int_array[...,None] >> np.arange(n - 1, -1, -1)) & 1
    ).astype(np.uint8)


def get_segments(sequences, segment_size, startpos=0, stride=None):
    """Splice a given set of sequences into segments and return indices.
    """
    if np.ndim(sequences) == 1:
        sequences = sequences[None,:]
    if stride is None:
        stride = segment_size
    nseqs, nbases = sequences.shape
    starts = np.arange(startpos, nbases, stride)
    if starts[-1] + segment_size > nbases:
        starts = starts[:-1]
    segments = np.array(
        [np.arange(startidx, startidx + segment_size) for startidx in starts]
    )
    return segments
