"""General helper functions with JAX acceleration

"""

import jax.numpy as jnp
import numpy as np


def binary_arr_to_int(bin_arr):
    """Convert one or more binary arrays to integer value(s).
    """
    k = bin_arr.shape[-1]
    weights = 1 << jnp.arange(k)[::-1]
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

def get_nested_depth(x):
    """Check the depth of a nested data structure, e.g. a list of lists.

    Args:
        x (np.ndarray | list | tuple | float): Data to check depth
    
    Returns:
        (int) : Number of layers n in the data x, so that the first item in x
            can be accessed via x[0][0]...[0] with n array accesses.
    """
    if np.isscalar(x) or jnp.isscalar(x):
        return 0  # base case, x is a single item
    # if isinstance(x, (np.ndarray, list, tuple)):
        # return 1 + max([get_nested_depth(item[0]) for item in x])
    return 1 + get_nested_depth(x[0])
