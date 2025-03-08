"""Core functions

"""

import math
import numpy as np

from .helpers import get_segments, int_to_binary_arr


def compare_sequences(seq1, seq2):
    """Screen for all differing positions between seq1 and seq2.
    Supports broadcasting.
    """
    seq1 = np.broadcast_to(seq1, np.broadcast_shapes(seq1.shape, seq2.shape))
    seq2 = np.broadcast_to(seq2, np.broadcast_shapes(seq1.shape, seq2.shape))
    return seq1 != seq2


def count_mutations(seqs, wt_seq):
    """Count the number of mutations in a sequence relative to another.
    """
    mut_screen = compare_sequences(seqs, wt_seq)
    nmuts = np.sum(mut_screen, axis=-1)
    return nmuts


def compute_mean_wildtype_expression(
        sequences,
        expression,
        wt_sequence,
):
    """Mean expression when base pair j is the wildtype.
    """
    mut_screen = compare_sequences(sequences, wt_sequence)
    wt_screen = ~mut_screen
    num_wts = wt_screen.sum(axis=0)  # number of observed wildtype bases
    wt_counts = wt_screen * expression[:,None]
    mu_mean = wt_counts.sum(axis=0) / num_wts
    assert mu_mean.shape == (len(wt_sequence),), "Bad shape"
    return mu_mean


def compute_mutualinfo_mutation_vs_expression_shift(
        sequences, 
        expression,
        wt_seq,
):
    """Mutual information by base pair of mutation status and expression shift.
    """
    wt_mean_expression = compute_mean_wildtype_expression(
        sequences, expression, wt_seq
    )

    nseqs, nbases = sequences.shape
    mutation_screen = compare_sequences(sequences, wt_seq)
    wt_screen = ~mutation_screen
    
    # screen[i,j] asserts that expression level associated with of sequence i 
    # is greater than the mean expression level when position j is WT.
    increase_exp_screen = expression[:,None] > wt_mean_expression[None,:]
    assert increase_exp_screen.shape == (nseqs, nbases)
    
    # p_i[j,k] with j mutation status, k shift
    p = np.zeros([nbases, 2, 2])
    
    # No mutation, reduction in expression
    p[:,0,0] = np.sum(wt_screen & (~increase_exp_screen), axis=0)
    # No mutation, increase in expression
    p[:,0,1] = np.sum(wt_screen & increase_exp_screen, axis=0)
    # Mutation, reduction in expression
    p[:,1,0] = np.sum(mutation_screen & (~increase_exp_screen), axis=0)
    # Mutation, increase in expression
    p[:,1,1] = np.sum(mutation_screen & increase_exp_screen, axis=0)
    
    p /= nseqs
    assert np.allclose(p.sum(axis=(1,2)), 1), "Probabilities should sum to 1."
    
    # Marginal distributions
    p_marg_mut = p.sum(axis=2)
    p_marg_exp = p.sum(axis=1)

    mut_info = np.zeros(nbases)
    for j in range(2):  # loop over possible mutation status
        for k in range(2):  # loop over possible expression status
            mut_info += p[:,j,k] * np.log2(
                p[:,j,k] / (p_marg_mut[:,j] * p_marg_exp[:,k])
            )

    return mut_info, p


def compute_mean_expression_shift(
        sequences,
        expression,
        wt_seq,
):
    """Compute the average change in expression level resulting from a mutation.

    The expression shift at position j of a given sequence is the difference
    between the expression level associated with that sequence and the average
    expression level across all sequences with the wild type at position j.
    The mean expression shift if the average of this quantity, computed across
    all sequences with a mutation at position j.

    """
    wt_mean_expression = compute_mean_wildtype_expression(
        sequences, expression, wt_seq
    )

    nseqs, nbases = sequences.shape
    mut_screen = compare_sequences(sequences, wt_seq)
    
    # shifts[i,j] is the difference between expression associated with sequence i 
    # and the mean expression level when position j is WT.
    shifts = expression[:,None] - wt_mean_expression[None,:]
    assert shifts.shape == (nseqs, nbases)
    
    # Want to average shift values over only the mutations at each base
    mean_expression_shift = (mut_screen * shifts).sum(0) / mut_screen.sum(0)

    return mean_expression_shift


def compute_segmented_mean_expression(
        sequences,
        expression,
        wt_seq,
        segment_size,
):
    """Compute the average expression level resulting from a mutation,
    across segments of a fixed size.

    """
    nseqs, nbases = sequences.shape
    mut_screen = compare_sequences(sequences, wt_seq)

    segments = get_segments(
        sequences, segment_size, 
        startpos=0, 
        stride=segment_size
    )

    nsegments = segments.shape[0]

    mut_screen_over_segments = np.array(
        [mut_screen[i,segments] for i in range(nseqs)]
    )

    # Each length k segment's binary string mutation profile corresponds to an 
    # index in [0, 2**k).
    nidxs = 2**segment_size
    weights = 1 << np.arange(segment_size)[::-1]
    mutation_profiles = mut_screen_over_segments @ weights

    # We now need to loop over the segments, and compute the expression.
    mean_exp_by_index = np.zeros([nidxs, nsegments])
    for mutidx in range(nidxs):
        idx_screen = mutation_profiles == mutidx
        mean_exp_by_index[mutidx] = np.sum(
            idx_screen * expression[:,None],
            axis=0
        ) / idx_screen.sum(0)

    return mean_exp_by_index


def compute_gamma(sequences, expression, wt_seq, segment_size=2):
    """TODO: Describe"""
    mean_exp_by_index = compute_segmented_mean_expression(
        sequences, expression, wt_seq, segment_size
    )
    xi = mean_exp_by_index - mean_exp_by_index[0]

    gamma = np.zeros([1 + segment_size, xi.shape[1]])
    mutation_profile_map = {
        i: int_to_binary_arr(i, segment_size) for i in range(1, xi.shape[0])
    }
    for idx in mutation_profile_map:
        profile = mutation_profile_map[idx]
        prof_nmuts = np.sum(profile)
        gamma[prof_nmuts] += xi[idx]**2 / math.comb(segment_size, prof_nmuts)
    return gamma


def compute_pairwise_segmented_mean_expression(
        sequences,
        expression,
        wt_seq,
        segment_size,
):
    """Compute the pairwise average expression level resulting from a mutation,
    across segments of a fixed size.

    """
    nseqs, nbases = sequences.shape
    mut_screen = compare_sequences(sequences, wt_seq)

    segments = get_segments(
        sequences, segment_size, 
        startpos=0, 
        stride=segment_size
    )

    nsegments = segments.shape[0]

    mut_screen_over_segments = np.array(
        [mut_screen[i,segments] for i in range(nseqs)]
    )

    # Each length k segment's binary string mutation profile corresponds to an 
    # index in [0, 2**k).
    nidxs = 2**segment_size
    weights = 1 << np.arange(segment_size)[::-1]
    mutation_profiles = mut_screen_over_segments @ weights

    # We now need to loop over the segments, and compute the expression.
    mean_exp_by_index = np.zeros([nidxs, nidxs, nsegments, nsegments])
    for mutidx1 in range(nidxs):
        idx_screen1 = mutation_profiles == mutidx1
        for mutidx2 in range(nidxs):
            idx_screen2 = mutation_profiles == mutidx2
            joint_screen = np.bitwise_and(
                idx_screen1[:, :, None], 
                idx_screen2[:, None, :]
            )
            exp_levels = joint_screen * expression[:,None,None]
            mean_exp_by_index[mutidx1, mutidx2,:,:] = np.sum(
                exp_levels,
                axis=0
            ) / joint_screen.sum(0)

    return mean_exp_by_index
