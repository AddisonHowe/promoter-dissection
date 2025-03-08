"""Tests for core functions.

"""

import pytest
import numpy as np

from promdis.helpers import binary_arr_to_int, int_to_binary_arr
from promdis.core import compare_sequences, count_mutations, get_segments
from promdis.core import compute_mean_wildtype_expression
from promdis.core import compute_segmented_mean_expression
from promdis.core import compute_pairwise_segmented_mean_expression

NA = np.nan

@pytest.mark.parametrize("seq1, seq2, expected", [
    [[0,3,0,0], [0,3,0,1], [0,0,0,1]],
    [[0,0,2,0], [1,0,2,0], [1,0,0,0]],
    [[2,2,2,1], [0,0,0,1], [1,1,1,0]],
    [[[1,1,2,1],[0,0,2,1]], [0,0,0,1], [[1,1,1,0],[0,0,1,0]]],
])
def test_compare_sequences(seq1, seq2, expected):
    seq1 = np.array(seq1)
    seq2 = np.array(seq2)
    val = compare_sequences(seq1, seq2)
    assert np.all(val == expected), f"Got:\n{val}\nExpected:\n{expected}"


@pytest.mark.parametrize("seqs, wt_seq, expected", [
    [[0,3,0,0], [0,3,0,1], 1],
    [[0,0,2,0], [1,0,2,0], 1],
    [[2,2,2,1], [0,0,0,1], 3],
    [[[1,1,2,1],[0,0,2,1]], [0,0,0,1], [3,1]],
])
def test_count_mutations(seqs, wt_seq, expected):
    seqs = np.array(seqs)
    wt_seq = np.array(wt_seq)
    val = count_mutations(seqs, wt_seq)
    assert np.all(val == expected), f"Got:\n{val}\nExpected:\n{expected}"


@pytest.mark.filterwarnings("ignore:invalid value encountered in divide")
@pytest.mark.parametrize("seqs, expression, wt_seq, exp_val", [
    [[[0,1,2,3],[3,3,3,3]], [1, 2], [0,0,3,3], [1,np.nan,2,1.5]],
    [[[0,1,2,3],[3,3,3,3]], [1, 3], [0,1,2,3], [1,1,1,2]],
])
def test_compute_mean_wildtype_expression(seqs, expression, wt_seq, exp_val):
    seqs = np.array(seqs)
    expression = np.array(expression)
    wt_seq = np.array(wt_seq)
    val = compute_mean_wildtype_expression(seqs, expression, wt_seq)
    errors = []
    if not (np.allclose(val, exp_val, equal_nan=True)):
        msg = f"Wrong value. Got:\n{val}\nExpected:\n{exp_val}"
        errors.append(msg)
    assert not errors, "Errors occurred:\n{}".format("\n".join(errors))
    

@pytest.mark.skip()
def test_compute_mutualinfo_mutation_vs_expression_shift(
    sequences, expression, wt_seq,
):
    raise NotImplementedError("Test not implemented!")
    

@pytest.mark.parametrize("bin_arr, expected", [
    [[0, 1, 0], 2],
    [[0, 1, 1], 3],
    [[1, 0, 1], 5],
    [[[0, 1, 0],[0, 1, 1],[1, 0, 1]], [2,3,5]],
    [[0], 0],
    [[1], 1],
])
def test_binary_arr_to_int(bin_arr, expected):
    val = binary_arr_to_int(np.array(bin_arr))
    assert np.all(val == expected), f"Got:\n{val}\nExpected:\n{expected}"


@pytest.mark.parametrize("int_array, n, expected", [
    [0, None, [0]],
    [1, None, [1]],
    [2, None, [1, 0]],
    [0, 1, [0]],
    [1, 1, [1]],
    [2, 2, [1, 0]],
    [0, 3, [0, 0, 0]],
    [1, 3, [0, 0, 1]],
    [2, 3, [0, 1, 0]],
])
def test_int_to_binary_arr(int_array, n, expected):
    val = int_to_binary_arr(int_array, n)
    assert np.all(val == expected), f"Got:\n{val}\nExpected:\n{expected}"


@pytest.mark.parametrize(
        "seqs,segment_size,startpos,stride,exp_val,exp_shape", [
    [[9,8,7,6,5,4,3,2,1,0], 2, 0, None, 
     [[0,1],[2,3],[4,5],[6,7],[8,9]], (5,2)],
    [[[8,7,6,5],[4,3,2,1]], 2, 0, None, 
     [[0,1],[2,3]], (2,2)],
    [[[8,7,6,5],[4,3,2,1]], 3, 0, None, 
     [[0,1,2]], (1,3)],
    [[[8,7,6,5],[4,3,2,1]], 3, 1, None, 
     [[1,2,3]], (1,3)],
    [[[8,7,6,5],[4,3,2,1]], 2, 1, 1, 
     [[1,2],[2,3]], (2,2)],
])
def test_get_segments(seqs, segment_size, startpos, stride, exp_val, exp_shape):
    seqs = np.array(seqs)
    val = get_segments(seqs, segment_size, startpos, stride)
    errors = []
    if not (val.shape == exp_shape):
        msg = f"Wrong shape. Got: {val.shape}\nExpected: {exp_shape}"
        errors.append(msg)
    if not (np.all(val == exp_val)):
        msg = f"Wrong values. Got:\n{val}\nExpected:\n{exp_val}"
        errors.append(msg)
    assert not errors, "Errors occurred:\n{}".format("\n".join(errors))


@pytest.mark.filterwarnings("ignore:invalid value encountered in divide")
@pytest.mark.parametrize(
    "seqs, expression, wt_seq, segment_size, exp_val, exp_shape", [
    [[[1,1,2,2,3,3,4,4],[1,2,2,1,3,4,3,4],[1,4,2,3,3,3,3,4]], 
     [2,4,2], [2,2,2,2,3,3,3,3], 2, 
     [[np.nan,2,2,np.nan],
      [np.nan,3,4,3],
      [4,np.nan,np.nan,np.nan],
      [2,np.nan,np.nan,2]],
     (4,4)],
])
def test_compute_segmented_mean_expression(
    seqs, expression, wt_seq, segment_size, exp_val, exp_shape
):
    seqs = np.array(seqs)
    expression = np.array(expression)
    wt_seq = np.array(wt_seq)
    val = compute_segmented_mean_expression(
        seqs, expression, wt_seq, segment_size
    )
    errors = []
    if not (np.allclose(val, exp_val, equal_nan=True)):
        msg = f"Wrong value. Got:\n{val}\nExpected:\n{exp_val}"
        errors.append(msg)
    if not (val.shape == exp_shape):
        msg = f"Wrong shape. Got: {val.shape} Expected: {exp_shape}"
        errors.append(msg)
    assert not errors, "Errors occurred:\n{}".format("\n".join(errors))


@pytest.mark.filterwarnings("ignore:invalid value encountered in divide")
@pytest.mark.parametrize(
    "seqs, expression, wt_seq, segment_size, exp_val, exp_shape", [
    [[[1,1,2,2,3,3,4,4],[1,2,2,1,3,4,3,4],[1,4,2,3,3,3,3,4]], 
     [2,4,2], [2,2,2,2,3,3,3,3], 2, 
     [[[[NA,NA,NA,NA],[NA,2,2,NA],[NA,2,2,NA],[NA,NA,NA,NA]],      # (0,0) vs (0,0)
       [[NA,NA,NA,NA],[NA,NA,NA,NA],[NA,2,NA,2],[NA,NA,NA,NA]],    # (0,0) vs (0,1)
       [[NA,NA,NA,NA],[NA,NA,NA,NA],[NA,NA,NA,NA],[NA,NA,NA,NA]],  # (0,0) vs (1,0)
       [[NA,NA,NA,NA],[2,NA,NA,2],[2,NA,NA,2],[NA,NA,NA,NA]]],     # (0,0) vs (1,1)
      [[[NA,NA,NA,NA],[NA,NA,2,NA],[NA,NA,NA,NA],[NA,NA,2,NA]],    # (0,1) vs (0,0)
       [[NA,NA,NA,NA],[NA,3,4,3],[NA,4,4,4],[NA,3,4,3]],           # (0,1) vs (0,1)
       [[NA,NA,NA,NA],[4,NA,NA,NA],[4,NA,NA,NA],[4,NA,NA,NA]],     # (0,1) vs (1,0)
       [[NA,NA,NA,NA],[2,NA,NA,NA],[NA,NA,NA,NA],[2,NA,NA,NA]]],   # (0,1) vs (1,1)
      [[[NA,NA,NA,NA],[NA,NA,NA,NA],[NA,NA,NA,NA],[NA,NA,NA,NA]],  # (1,0) vs (0,0)
       [[NA,4,4,4],[NA,NA,NA,NA],[NA,NA,NA,NA],[NA,NA,NA,NA]],     # (1,0) vs (0,1)
       [[4,NA,NA,NA],[NA,NA,NA,NA],[NA,NA,NA,NA],[NA,NA,NA,NA]],   # (1,0) vs (1,0)
       [[NA,NA,NA,NA],[NA,NA,NA,NA],[NA,NA,NA,NA],[NA,NA,NA,NA]]], # (1,0) vs (1,1)
      [[[NA,2,2,NA],[NA,NA,NA,NA],[NA,NA,NA,NA],[NA,2,2,NA]],      # (1,1) vs (0,0)
       [[NA,2,NA,2],[NA,NA,NA,NA],[NA,NA,NA,NA],[NA,NA,NA,NA]],    # (1,1) vs (0,1)
       [[NA,NA,NA,NA],[NA,NA,NA,NA],[NA,NA,NA,NA],[NA,NA,NA,NA]],  # (1,1) vs (1,0)
       [[2,NA,NA,2],[NA,NA,NA,NA],[NA,NA,NA,NA],[2,NA,NA,2]]]],    # (1,1) vs (1,1)
     (4,4,4,4)],
])
def test_compute_pairwise_segmented_mean_expression(
    seqs, expression, wt_seq, segment_size, exp_val, exp_shape
):
    seqs = np.array(seqs)
    expression = np.array(expression)
    wt_seq = np.array(wt_seq)
    val = compute_pairwise_segmented_mean_expression(
        seqs, expression, wt_seq, segment_size
    )
    exp_val = np.array(exp_val)
    errors = []
    if not (np.allclose(val, exp_val, equal_nan=True)):
        msg = f"Wrong value. Got:\n{val}\nExpected:\n{exp_val}"
        # msg = f"{val != exp_val}"
        errors.append(msg)
    if not (val.shape == exp_shape):
        msg = f"Wrong shape. Got: {val.shape} Expected: {exp_shape}"
        errors.append(msg)
    assert not errors, "Errors occurred:\n{}".format("\n".join(errors))


@pytest.mark.skip()
def test_compute_gamma(sequences, expression, wt_seq, segment_size):
    raise NotImplementedError("Test not implemented!")
    assert np.all(val == expected), f"Got:\n{val}\nExpected:\n{expected}"
