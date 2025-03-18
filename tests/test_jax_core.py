"""Tests for core functions.

"""

import pytest
import numpy as np

from promdis.jax.helpers import binary_arr_to_int, int_to_binary_arr, get_segments
from promdis.jax.core import compare_sequences, count_mutations
from promdis.jax.core import compute_mean_wildtype_expression
from promdis.jax.core import compute_mean_expression_shift
from promdis.jax.core import compute_total_expression_by_mutation
from promdis.jax.core import compute_mean_expression_by_mutation
from promdis.jax.core import compute_expression_shift_by_mutation
from promdis.jax.core import compute_total_expression_by_pairwise_mutation
from promdis.jax.core import compute_mean_expression_by_pairwise_mutation
from promdis.jax.core import compute_expression_shift_by_pairwise_mutation

NA = np.nan


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
    assert np.allclose(val, expected), f"Got:\n{val}\nExpected:\n{expected}"


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
    assert np.allclose(val, expected), f"Got:\n{val}\nExpected:\n{expected}"


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
    if not (np.allclose(val, exp_val)):
        msg = f"Wrong values. Got:\n{val}\nExpected:\n{exp_val}"
        errors.append(msg)
    assert not errors, "Errors occurred:\n{}".format("\n".join(errors))


@pytest.mark.parametrize("seq1, seq2, expected", [
    [[0,3,0,0], [0,3,0,1], [0,0,0,1]],
    [[0,0,2,0], [1,0,2,0], [1,0,0,0]],
    [[2,2,2,1], [0,0,0,1], [1,1,1,0]],
    [[[1,1,2,1],[0,0,2,1]], [0,0,0,1], [[1,1,1,0],[0,0,1,0]]],
    [[0,0,0,1], [[1,1,2,1],[0,0,2,1]], [[1,1,1,0],[0,0,1,0]]],
])
def test_compare_sequences(seq1, seq2, expected):
    seq1 = np.array(seq1)
    seq2 = np.array(seq2)
    val = compare_sequences(seq1, seq2)
    assert np.allclose(val, expected), f"Got:\n{val}\nExpected:\n{expected}"


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
    assert np.allclose(val, expected), f"Got:\n{val}\nExpected:\n{expected}"


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
def test_compute_mean_expression_shift(
    sequences, expression, wt_seq,
):
    raise NotImplementedError("Test not implemented!")


##############################################
##  Test single segment mutation functions  ##
##############################################

@pytest.mark.parametrize(
    "seqs, expression, wt_seq, segment_size, " + \
    "exp_total_expression, exp_total_expression_shape, exp_counts, exp_counts_shape,", [
    [[[1,1,2,2,3,3,4,4],[1,2,2,1,3,4,3,4],[1,4,2,3,3,3,3,4]], 
     [2,4,2], [2,2,2,2,3,3,3,3], 2, 
     # Expression
     [[0,2,4,0],
      [0,6,4,6],
      [4,0,0,0],
      [4,0,0,2],],
     (4,4),
     # Counts
     [[0,1,2,0],
      [0,2,1,2],
      [1,0,0,0],
      [2,0,0,1],],
     (4,4),
    ],
])
class TestSingleSegmentMutationFunctions:
    
    def test_compute_total_expression_by_mutation(
        self, seqs, expression, wt_seq, segment_size, 
        exp_total_expression, exp_total_expression_shape,
        exp_counts, exp_counts_shape,
    ):
        seqs = np.array(seqs)
        expression = np.array(expression)
        wt_seq = np.array(wt_seq)
        exp_total_expression = np.array(exp_total_expression)
        exp_counts = np.array(exp_counts)
        total_expression, counts = compute_total_expression_by_mutation(
            seqs, expression, wt_seq, segment_size
        )
        errors = []
        if not (total_expression.shape == exp_total_expression_shape):
            msg = f"Wrong shape for total_expression. " + \
                f"Got: {total_expression.shape} Expected: {exp_total_expression_shape}"
            errors.append(msg)
        if not (counts.shape == exp_counts_shape):
            msg = f"Wrong shape for counts. " + \
                f"Got: {counts.shape} Expected: {exp_counts_shape}"
            errors.append(msg)
        if not (np.allclose(total_expression, exp_total_expression, equal_nan=True)):
            msg = f"Error in total_expression. " \
                + f"Got:\n{total_expression}\nExpected:\n{exp_total_expression}"
            errors.append(msg)
        if not (np.allclose(counts, exp_counts, equal_nan=True)):
            msg = f"Error in counts. Got:\n{counts}\nExpected:\n{exp_counts}"
            errors.append(msg)
        assert not errors, "Errors occurred:\n{}".format("\n".join(errors))

    @pytest.mark.filterwarnings("ignore:invalid value encountered in divide")
    def test_compute_mean_expression_by_mutation(
        self, seqs, expression, wt_seq, segment_size, 
        exp_total_expression, exp_total_expression_shape,
        exp_counts, exp_counts_shape,
    ):
        seqs = np.array(seqs)
        expression = np.array(expression)
        wt_seq = np.array(wt_seq)
        exp_total_expression = np.array(exp_total_expression)
        exp_counts = np.array(exp_counts)
        exp_mean_expression = exp_total_expression / exp_counts
        mean_expression = compute_mean_expression_by_mutation(
            seqs, expression, wt_seq, segment_size
        )
        errors = []
        if not (mean_expression.shape == exp_mean_expression.shape):
            msg = f"Wrong shape for expression. " + \
                f"Got: {mean_expression.shape} Expected: {exp_mean_expression.shape}"
            errors.append(msg)
        if not (np.allclose(mean_expression, exp_mean_expression, equal_nan=True)):
            msg = f"Error in mean expression. " + \
                f"Got:\n{mean_expression}\nExpected:\n{exp_mean_expression}"
            errors.append(msg)
        assert not errors, "Errors occurred:\n{}".format("\n".join(errors))


@pytest.mark.filterwarnings("ignore:invalid value encountered in divide")
@pytest.mark.parametrize(
    "sequences, expression, wt_seq, segment_size, profile_groups, exp_xi", [
    [[[1,1,2,2,3,3,4,4],[1,2,2,1,3,4,3,4],[1,4,2,3,3,3,3,4]], 
     [2,4,2], [2,2,2,2,3,3,3,3], 2, 
     None,  # Profile Groups: [[(0, 0)], [(0, 1)], [(1, 0)], [(1, 1)]]
     [[NA,0,0,NA],      # xi[(0,0)]
      [NA,1,2,NA],     # xi[(0,1)]
      [NA,NA,NA,NA],  # xi[(1,0)]
      [NA,NA,NA,NA]], # xi[(1,1)]
    ],
    [[[1,1,2,2,3,3,4,4],[1,2,2,1,3,4,3,4],[1,4,2,3,3,3,3,4]], 
     [2,4,2], [2,2,2,2,3,3,3,3], 2, 
     'nmuts',  # Profile Groups: [[(0, 0)], [(0, 1), (1, 0)], [(1, 1)]]
     [[NA,0,0,NA],      # xi[(0,0)]
      [NA,1,2,NA],      # xi[{(0,1),(1,0)}]
      [NA,NA,NA,NA]],   # xi[(1,1)]
    ],
    [[[1,1,2,2,3,3,4,4],[1,2,2,1,3,4,3,4],[1,4,2,3,3,3,3,4]], 
     [2,4,2], [2,2,2,2,3,3,3,3], 2, 
     [[(0,0)],[(0,1),(1,0)],[(1,1)]],  # Profile Groups
     [[NA,0,0,NA],      # xi[(0,0)]
      [NA,1,2,NA],      # xi[{(0,1),(1,0)}]
      [NA,NA,NA,NA]],   # xi[(1,1)]
    ],
    [[[1,1,2,2,3,3,4,4],[1,2,2,1,3,4,3,4],[1,4,2,3,3,3,3,4]], 
     [2,4,2], [2,2,2,2,3,3,3,3], 2, 
     [[(0,0),(0,1),(1,0),(1,1)]],  # Profile Groups
     [[NA,2/3,2/3,NA]], # xi[{(0, 0),(0, 1),(1, 0),(1, 1)}]
    ],
    [[[1,1,2,2,3,3,4,4],[1,2,2,1,3,4,3,4],[1,4,2,3,3,3,3,4]], 
     [2,4,2], [2,2,2,2,3,3,3,3], 2, 
     [(0,0),(0,1),(1,0),(1,1)],  # Profile Groups (as depth-1 list)
     [NA,2/3,2/3,NA], # xi[{(0, 0),(0, 1),(1, 0),(1, 1)}]
    ],
    [[[1,1,2,2,3,3,4,4],[1,2,2,1,3,4,3,4],[1,4,2,3,3,3,3,4]], 
     [2,4,2], [2,2,2,2,3,3,3,3], 2, 
     (0,1),  # Profile Groups: (0, 1)
     [NA,1,2,NA],     # xi[(0,1)]
    ],
])
def test_compute_expression_shift_by_mutation(
    sequences, expression, wt_seq, segment_size, profile_groups, 
    exp_xi,
):
    sequences = np.array(sequences)
    expression = np.array(expression)
    wt_seq = np.array(wt_seq)
    exp_xi = np.array(exp_xi)
    xi, profiles = compute_expression_shift_by_mutation(
        sequences, expression, wt_seq, 
        segment_size=segment_size, 
        profile_groups=profile_groups,
    )

    errors = []
    if not np.allclose(xi, exp_xi, equal_nan=True):
        msg = f"Error in xi. Got:\n{xi}\nExpected:\n{exp_xi}"
        errors.append(msg)
    assert not errors, "Errors occurred:\n{}".format("\n".join(errors))


################################################
##  Test pairwise segment mutation functions  ##
################################################

@pytest.mark.parametrize(
    "seqs, expression, wt_seq, segment_size, " + \
    "exp_total_expression, exp_total_expression_shape, exp_counts, exp_counts_shape,", [
    [[[1,1,2,2,3,3,4,4],[1,2,2,1,3,4,3,4],[1,4,2,3,3,3,3,4]], 
     [2,4,2], [2,2,2,2,3,3,3,3], 2, 
     # Total Expression
     [[[[0,0,0,0],[0,2,2,0],[0,2,4,0],[0,0,0,0]],    # (0,0) vs (0,0)
       [[0,0,0,0],[0,0,0,0],[0,2,0,2],[0,0,0,0]],    # (0,0) vs (0,1)
       [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]],    # (0,0) vs (1,0)
       [[0,0,0,0],[2,0,0,2],[4,0,0,2],[0,0,0,0]]],   # (0,0) vs (1,1)
      [[[0,0,0,0],[0,0,2,0],[0,0,0,0],[0,0,2,0]],    # (0,1) vs (0,0)
       [[0,0,0,0],[0,6,4,6],[0,4,4,4],[0,6,4,6]],    # (0,1) vs (0,1)
       [[0,0,0,0],[4,0,0,0],[4,0,0,0],[4,0,0,0]],    # (0,1) vs (1,0)
       [[0,0,0,0],[2,0,0,0],[0,0,0,0],[2,0,0,0]]],   # (0,1) vs (1,1)
      [[[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]],    # (1,0) vs (0,0)
       [[0,4,4,4],[0,0,0,0],[0,0,0,0],[0,0,0,0]],    # (1,0) vs (0,1)
       [[4,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]],    # (1,0) vs (1,0)
       [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]],   # (1,0) vs (1,1)
      [[[0,2,4,0],[0,0,0,0],[0,0,0,0],[0,2,2,0]],    # (1,1) vs (0,0)
       [[0,2,0,2],[0,0,0,0],[0,0,0,0],[0,0,0,0]],    # (1,1) vs (0,1)
       [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]],    # (1,1) vs (1,0)
       [[4,0,0,2],[0,0,0,0],[0,0,0,0],[2,0,0,2]]]],  # (1,1) vs (1,1)
     (4,4,4,4),
     # Counts
     [[[[0,0,0,0],[0,1,1,0],[0,1,2,0],[0,0,0,0]],    # (0,0) vs (0,0)
       [[0,0,0,0],[0,0,0,0],[0,1,0,1],[0,0,0,0]],    # (0,0) vs (0,1)
       [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]],    # (0,0) vs (1,0)
       [[0,0,0,0],[1,0,0,1],[2,0,0,1],[0,0,0,0]]],   # (0,0) vs (1,1)
      [[[0,0,0,0],[0,0,1,0],[0,0,0,0],[0,0,1,0]],    # (0,1) vs (0,0)
       [[0,0,0,0],[0,2,1,2],[0,1,1,1],[0,2,1,2]],    # (0,1) vs (0,1)
       [[0,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0]],    # (0,1) vs (1,0)
       [[0,0,0,0],[1,0,0,0],[0,0,0,0],[1,0,0,0]]],   # (0,1) vs (1,1)
      [[[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]],    # (1,0) vs (0,0)
       [[0,1,1,1],[0,0,0,0],[0,0,0,0],[0,0,0,0]],    # (1,0) vs (0,1)
       [[1,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]],    # (1,0) vs (1,0)
       [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]],   # (1,0) vs (1,1)
      [[[0,1,2,0],[0,0,0,0],[0,0,0,0],[0,1,1,0]],    # (1,1) vs (0,0)
       [[0,1,0,1],[0,0,0,0],[0,0,0,0],[0,0,0,0]],    # (1,1) vs (0,1)
       [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]],    # (1,1) vs (1,0)
       [[2,0,0,1],[0,0,0,0],[0,0,0,0],[1,0,0,1]]]],  # (1,1) vs (1,1)
     (4,4,4,4)
    ],
])
class TestPairwiseSegmentMutationFunctions:
    
    def test_compute_total_expression_by_pairwise_mutation(
        self, seqs, expression, wt_seq, segment_size, 
        exp_total_expression, exp_total_expression_shape,
        exp_counts, exp_counts_shape,
    ):
        seqs = np.array(seqs)
        expression = np.array(expression)
        wt_seq = np.array(wt_seq)
        exp_total_expression = np.array(exp_total_expression)
        exp_counts = np.array(exp_counts)
        total_expression, counts = compute_total_expression_by_pairwise_mutation(
            seqs, expression, wt_seq, segment_size
        )
        errors = []
        if not (total_expression.shape == exp_total_expression_shape):
            msg = f"Wrong shape for total expression. " + \
                f"Got: {total_expression.shape} Expected: {exp_total_expression_shape}"
            errors.append(msg)
        if not (counts.shape == exp_counts_shape):
            msg = f"Wrong shape for counts. " + \
                f"Got: {counts.shape} Expected: {exp_counts_shape}"
            errors.append(msg)
        if not np.allclose(total_expression, exp_total_expression, equal_nan=True):
            msg = f"Error in total expression. " + \
                f"Got:\n{total_expression}\nExpected:\n{exp_total_expression}"
            errors.append(msg)
        if not np.allclose(counts, exp_counts, equal_nan=True):
            msg = f"Error in counts. Got:\n{counts}\nExpected:\n{exp_counts}"
            errors.append(msg)
        assert not errors, "Errors occurred:\n{}".format("\n".join(errors))

    @pytest.mark.filterwarnings("ignore:invalid value encountered in divide")
    def test_compute_mean_expression_by_pairwise_mutation(
        self, seqs, expression, wt_seq, segment_size, 
        exp_total_expression, exp_total_expression_shape,
        exp_counts, exp_counts_shape,
    ):
        seqs = np.array(seqs)
        expression = np.array(expression)
        wt_seq = np.array(wt_seq)
        exp_total_expression = np.array(exp_total_expression)
        exp_counts = np.array(exp_counts)
        exp_mean_expression = exp_total_expression / exp_counts
        mean_expression = compute_mean_expression_by_pairwise_mutation(
            seqs, expression, wt_seq, segment_size
        )
        errors = []
        if not (mean_expression.shape == exp_total_expression_shape):
            msg = f"Wrong shape for mean expression. " + \
                f"Got: {mean_expression.shape} Expected: {exp_total_expression_shape}"
            errors.append(msg)
        if not (np.allclose(mean_expression, exp_mean_expression, equal_nan=True)):
            msg = f"Error in mean expression. " + \
                f"Got:\n{mean_expression}\nExpected:\n{exp_mean_expression}"
            errors.append(msg)
        assert not errors, "Errors occurred:\n{}".format("\n".join(errors))


@pytest.mark.filterwarnings("ignore:invalid value encountered in divide")
@pytest.mark.parametrize(
    "sequences, expression, wt_seq, segment_size, profile_groups, exp_xi", [
    [[[1,1,2,2,3,3,4,4],[1,2,2,1,3,4,3,4],[1,4,2,3,3,3,3,4]], 
     [2,4,2], [2,2,2,2,3,3,3,3], 2, 
     None,  # Profile Groups: [(0,0;0,0),(0,0;0,1),(0,0;1,0),(0,0;0,0)]
     [[[NA,NA,NA,NA],[NA,0,0,NA],[NA,0,0,NA],[NA,NA,NA,NA]],    # xi[(0,0;0,0)]
      [[NA,NA,NA,NA],[NA,NA,NA,NA],[NA,0,NA,NA],[NA,NA,NA,NA]],  # xi[(0,0;0,1)]
      [[NA,NA,NA,NA],[NA,NA,NA,NA],[NA,NA,NA,NA],[NA,NA,NA,NA]],    # xi[(0,0;1,0)]
      [[NA,NA,NA,NA],[NA,NA,NA,NA],[NA,NA,NA,NA],[NA,NA,NA,NA]],  # xi[(0,0;1,1)]
      [[NA,NA,NA,NA],[NA,NA,0,NA],[NA,NA,NA,NA],[NA,NA,NA,NA]],    # xi[(0,1;0,0)]
      [[NA,NA,NA,NA],[NA,1,2,NA],[NA,2,2,NA],[NA,NA,NA,NA]],    # xi[(0,1;0,1)]
      [[NA,NA,NA,NA],[NA,NA,NA,NA],[NA,NA,NA,NA],[NA,NA,NA,NA]],    # xi[(0,1;1,0)]
      [[NA,NA,NA,NA],[NA,NA,NA,NA],[NA,NA,NA,NA],[NA,NA,NA,NA]],  # xi[(0,1;1,1)]
      [[NA,NA,NA,NA],[NA,NA,NA,NA],[NA,NA,NA,NA],[NA,NA,NA,NA]],    # xi[(1,0;0,0)]
      [[NA,NA,NA,NA],[NA,NA,NA,NA],[NA,NA,NA,NA],[NA,NA,NA,NA]],    # xi[(1,0;0,1)]
      [[NA,NA,NA,NA],[NA,NA,NA,NA],[NA,NA,NA,NA],[NA,NA,NA,NA]],    # xi[(1,0;1,0)]
      [[NA,NA,NA,NA],[NA,NA,NA,NA],[NA,NA,NA,NA],[NA,NA,NA,NA]],  # xi[(1,0;1,1)]
      [[NA,NA,NA,NA],[NA,NA,NA,NA],[NA,NA,NA,NA],[NA,NA,NA,NA]],    # xi[(1,1;0,0)]
      [[NA,NA,NA,NA],[NA,NA,NA,NA],[NA,NA,NA,NA],[NA,NA,NA,NA]],    # xi[(1,1;0,1)]
      [[NA,NA,NA,NA],[NA,NA,NA,NA],[NA,NA,NA,NA],[NA,NA,NA,NA]],    # xi[(1,1;1,0)]
      [[NA,NA,NA,NA],[NA,NA,NA,NA],[NA,NA,NA,NA],[NA,NA,NA,NA]],  # xi[(1,1;1,1)]
    ]],
    [[[1,1,2,2,3,3,4,4],[1,2,2,1,3,4,3,4],[1,4,2,3,3,3,3,4]], 
     [2,4,2], [2,2,2,2,3,3,3,3], 2, 
     [((0,0),(0,1))],  # Profile Groups: [(0,0;0,0),(0,0;0,1),(0,0;1,0),(0,0;0,0)]
     [[NA,NA,NA,NA],[NA,NA,NA,NA],[NA,0,NA,NA],[NA,NA,NA,NA]],
    ],
])
def test_compute_expression_shift_by_pairwise_mutation(
    sequences, expression, wt_seq, segment_size, profile_groups, 
    exp_xi,
):
    sequences = np.array(sequences)
    expression = np.array(expression)
    wt_seq = np.array(wt_seq)
    exp_xi = np.array(exp_xi)
    xi, profiles = compute_expression_shift_by_pairwise_mutation(
        sequences, expression, wt_seq, 
        segment_size=segment_size, 
        profile_groups=profile_groups,
    )

    errors = []
    if not np.allclose(xi, exp_xi, equal_nan=True):
        msg = f"Error in xi. Got:\n{xi}\nExpected:\n{exp_xi}"
        errors.append(msg)
    assert not errors, "Errors occurred:\n{}".format("\n".join(errors))



#####################
##  Miscellaneous  ##
#####################
        
@pytest.mark.skip()
def test_compute_gamma(sequences, expression, wt_seq, segment_size):
    raise NotImplementedError("Test not implemented!")
    assert np.allclose(val, expected), f"Got:\n{val}\nExpected:\n{expected}"


@pytest.mark.skip()
def test_compute_mutualinfo_mutation_vs_expression_shift(
    sequences, expression, wt_seq,
):
    raise NotImplementedError("Test not implemented!")



    