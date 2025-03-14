"""Core functions

"""

import numpy as np

from .helpers import get_segments, int_to_binary_arr, binary_arr_to_int
from .helpers import get_nested_depth


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
        wt_seq,
):
    """Mean expression when base pair i is the wildtype.

    Args:
        sequences (np.ndarray[np.uint8]): Shape (nseqs, nbases).
        expression (np.ndarray[int]): Shape (nseqs,).
        wt_seq (np.ndarray[np.uint8]): Shape (nbases,)
    
    Returns:
        (np.ndarray[float]): Shape (nbases,). Array of mean expression values 
            such that the ith entry gives the mean expression across all
            sequences with the wildtype base at position i.
    """
    mut_screen = compare_sequences(sequences, wt_seq)
    wt_screen = ~mut_screen
    num_wts = wt_screen.sum(axis=0)  # number of observed wildtype bases
    wt_counts = wt_screen * expression[:,None]
    mu_mean = wt_counts.sum(axis=0) / num_wts
    assert mu_mean.shape == (len(wt_seq),), "Bad shape"
    return mu_mean


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

    Args:
        sequences (np.ndarray[np.uint8]): Shape (nseqs, nbases).
        expression (np.ndarray[int]): Shape (nseqs,).
        wt_seq (np.ndarray[np.uint8]): Shape (nbases,)
    
    Returns:
        (np.ndarray[float]): Shape (nbases,). Array of expression shift values
            such that the ith entry gives the difference between the mean 
            expression across all sequences with a mutation at position i, and 
            the mean expression level across all sequences with the wildtype 
            base at position i.
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
    total_expression = (mut_screen * shifts).sum(0)
    counts = mut_screen.sum(0)
    mean_expression_shift = total_expression / counts

    return mean_expression_shift


#########################################
##  Single segment mutation functions  ##
#########################################


def compute_total_expression_by_mutation(
        sequences,
        expression,
        wt_seq,
        segment_size=2,
):
    """Compute total expression and counts of each mutation per segment.

    Given segment_size k, returns two arrays of shape (2**k, L) where L is the 
    number of segments. The i-th row corresponds to the mutation profile given
    by the binary string representation of i.

    Args:
        sequences (np.ndarray[np.uint8]): Shape (nseqs, nbases).
        expression (np.ndarray[int]): Shape (nseqs,).
        wt_seq (np.ndarray[np.uint8]): Shape (nbases,)
        segment_size (int): Segment size.
    
    Returns:
        (np.ndarray): Shape (2**k, L). Total expression level. The (i,j) entry
            gives the sum of expression values across all sequences whose j-th
            segment has mutation profile BinaryRep(i).
        (np.ndarray): Shape (2**k, L). Counts. The (i,j) entry gives the total 
            number of sequences whose j-th segment has mutation profile 
            BinaryRep(i).
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

    total_expression_by_idx = np.zeros([nidxs, nsegments])
    counts_by_idx = np.zeros([nidxs, nsegments])
    for mutidx in range(nidxs):
        idx_screen = mutation_profiles == mutidx
        total_expression_by_idx[mutidx] = np.sum(
            idx_screen * expression[:,None], axis=0
        )
        counts_by_idx[mutidx] = idx_screen.sum(0)
    return total_expression_by_idx, counts_by_idx


def compute_mean_expression_by_mutation(
        sequences,
        expression,
        wt_seq,
        segment_size=2,
):
    """Mean expression by segment across all sequences with a certain mutation.

    Given segment_size k, returns array of shape (2**k, L) where L is the 
    number of segments. The i-th row corresponds to the mutation profile given
    by the binary string representation of i.

    Args:
        sequences (np.ndarray[np.uint8]): Shape (nseqs, nbases).
        expression (np.ndarray[int]): Shape (nseqs,).
        wt_seq (np.ndarray[np.uint8]): Shape (nbases,)
        segment_size (int): Segment size.
    
    Returns:
        (np.ndarray): Shape (2**k, L). Mean expression level. The (i,j) entry
            gives the average expression value across all sequences whose j-th
            segment has mutation profile BinaryRep(i).
    """
    tot_exp_by_idx, counts_by_idx = compute_total_expression_by_mutation(
        sequences, expression, wt_seq, segment_size
    )
    return tot_exp_by_idx / counts_by_idx


def compute_expression_shift_by_mutation(
        sequences,
        expression,
        wt_seq,
        segment_size=2,
        profile_groups=None,
):
    """Compute the expression shift resulting from a mutation.

    The expression shift is calculated as the difference between the mean 
    expression across sequences with a specified mutation at a given segment, 
    and the mean wildtype expression.

    Args:
        sequences (np.ndarray[np.uint8]): Shape (nseqs, nbases).
        expression (np.ndarray[int]): Shape (nseqs,).
        wt_seq (np.ndarray[np.uint8]): Shape (nbases,)
        segment_size (int): Segment size.
        profile_groups (list[list[tuple]] | str): Profile groups. Default None.
            If None, mean expression shift is calculated for each of the 2**k
            mutation profiles, where k is the specified segment size. If 'nmuts',
            the mean expression shift is calculated across groups of mutation 
            profiles, where each group is defined as the set of all mutation 
            profiles with a given number of mutations. Otherwise, may be specified 
            as a list of lists, where each sublist contains a number of binary 
            tuples, specifying those mutation profiles over which to compute the 
            mean expression shift.

    Returns:
        (np.ndarray): Shape (ngroups, L) or (L,). If a single profile group is 
            specified, either as a single tuple or a single list of tuples, the 
            returned array is the mean expression shift corresponding to that 
            profile. If multiple profile groups are given, each row of the 
            returned array corresponds to a profile group.
        (list[list[tuple]] | list[tuple]): Shape (ngroups, ?, k) or (?, K). 
            List of mutation profile groups (each itself a list), if multiple 
            groups are given. If a single profile group is specified, then a
            single list, specifying the profile. Each group is defined by 
            the list of mutation profiles, specified as a binary k-tuple.
    """
    # Get total expression and counts for each mutation profile.
    tot_exp_by_idx, counts_by_idx = compute_total_expression_by_mutation(
        sequences, expression, wt_seq, segment_size
    )
    nidxs, nsegments = tot_exp_by_idx.shape
    
    # Compute the mean expression level for the wildtype.
    mean_wildtype_expression = tot_exp_by_idx[0] / counts_by_idx[0]

    # Handle mutation profiles.
    if profile_groups is None:
        # If not specified, compute the shift for each mutation profile.
        nprofile_groups = nidxs
        profile_groups = [
            np.array([int_to_binary_arr(i, segment_size)]) 
            for i in range(nprofile_groups)
        ]
        nesting_depth = 3
    elif profile_groups == 'nmuts':
        # Group profiles by the total number of mutations [0, 1,..., k]
        nprofile_groups = 1 + segment_size
        profile_groups = [[] for _ in range(nprofile_groups)]
        for i in range(nidxs):
            profile = int_to_binary_arr(i, segment_size)
            nmuts = np.sum(profile)
            profile_groups[nmuts].append(profile)
        nesting_depth = 3
    else:
        # Otherwise, check that the given grouping of profiles is consistent.
        nesting_depth = get_nested_depth(profile_groups)
        if nesting_depth == 3:
            nprofile_groups = len(profile_groups)
        elif nesting_depth == 2:
            nprofile_groups = 1
            profile_groups = [profile_groups]
        elif nesting_depth == 1:
            nprofile_groups = 1
            profile_groups = [[profile_groups]]
            assert len(profile_groups)
        else:
            msg = f"Cannot handle given profile groups: {profile_groups}"
            raise RuntimeError(msg)
        profile_groups = [np.array(x) for x in profile_groups]
    
    total_expression = np.zeros([nprofile_groups, nsegments])
    counts = np.zeros([nprofile_groups, nsegments])
    for group_idx, profile_group in enumerate(profile_groups):
        for profile in profile_group:
            idx = binary_arr_to_int(profile)
            total_expression[group_idx] += tot_exp_by_idx[idx]
            counts[group_idx] += counts_by_idx[idx]
    
    mean_expression = total_expression / counts
    expression_shift = mean_expression - mean_wildtype_expression

    if nesting_depth < 3:
        return expression_shift[0], profile_groups[0]
    else:
        return expression_shift, profile_groups


###########################################
##  Pairwise segment mutation functions  ##
###########################################

def compute_total_expression_by_pairwise_mutation(
        sequences,
        expression,
        wt_seq,
        segment_size=2,
):
    """Compute total expression and counts for each pairwise mutation of segments.

    Given segment_size k, returns two arrays of shape (2**k, 2**k, L, L) where 
    L is the number of segments. The (i,j) entry corresponds to a pair of mutation 
    profiles, each given by the binary representation of i and j, respectively.

    Args:
        sequences (np.ndarray[np.uint8]): Shape (nseqs, nbases).
        expression (np.ndarray[int]): Shape (nseqs,).
        wt_seq (np.ndarray[np.uint8]): Shape (nbases,)
        segment_size (int): Segment size.
    
    Returns:
        (np.ndarray): Shape (2**k, 2**k, L, L). Total expression level. The 
            (i,j,k,l) entry gives the sum of expression values across all pairs
            of sequences s1 and s2 where the kth segment of s1 and the lth 
            segment of s2 have mutation profiles BinaryRep(i) and BinaryRep(j), 
            respectively.
        (np.ndarray): Shape (2**k, 2**k, L, L). Counts, analogously.
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

    total_expression_by_idx = np.zeros([nidxs, nidxs, nsegments, nsegments])
    counts_by_idx = np.zeros([nidxs, nidxs, nsegments, nsegments])
    for mutidx1 in range(nidxs):
        idx_screen1 = mutation_profiles == mutidx1
        for mutidx2 in range(nidxs):
            idx_screen2 = mutation_profiles == mutidx2
            joint_screen = np.bitwise_and(
                idx_screen1[:, :, None], 
                idx_screen2[:, None, :]
            )
            total_expression_by_idx[mutidx1,mutidx2,:,:] = np.sum(
                joint_screen * expression[:,None,None], axis=0
            )
            counts_by_idx[mutidx1,mutidx2] = joint_screen.sum(0)

    return total_expression_by_idx, counts_by_idx


def compute_mean_expression_by_pairwise_mutation(
        sequences,
        expression,
        wt_seq,
        segment_size=2,
):
    """Compute mean expression for each pairwise mutation of segments.

    Given segment_size k, returns an array of shape (2**k, 2**k, L, L) where 
    L is the number of segments. The (i,j) entry corresponds to a pair of mutation 
    profiles, each given by the binary representation of i and j, respectively.

    Args:
        sequences (np.ndarray[np.uint8]): Shape (nseqs, nbases).
        expression (np.ndarray[int]): Shape (nseqs,).
        wt_seq (np.ndarray[np.uint8]): Shape (nbases,)
        segment_size (int): Segment size.
    
    Returns:
        (np.ndarray): Shape (2**k, 2**k, L, L). Mean expression level. The 
            (i,j,k,l) entry gives the mean expression level across all pairs
            of sequences s1 and s2 where the kth segment of s1 and the lth 
            segment of s2 have mutation profiles BinaryRep(i) and BinaryRep(j), 
            respectively.
    """
    total_expression, counts = compute_total_expression_by_pairwise_mutation(
        sequences, expression, wt_seq, segment_size
    )
    return total_expression / counts


def compute_expression_shift_by_pairwise_mutation(
        sequences,
        expression,
        wt_seq,
        segment_size=2,
        profile_groups=None,
):
    """Compute the expression shift resulting from mutations to a pair of segments.

    The expression shift is calculated as the difference between the mean 
    expression across sequences with specified mutations at two segments, 
    and the mean wildtype expression. 

    Args:
        sequences (np.ndarray[np.uint8]): Shape (nseqs, nbases).
        expression (np.ndarray[int]): Shape (nseqs,).
        wt_seq (np.ndarray[np.uint8]): Shape (nbases,)
        segment_size (int): Segment size.
        profile_groups (list[list[tuple[tuple]]] | str): Profile groups. Default None.
            If None, mean expression shift is calculated for each of the 2*2**k
            mutation profiles, where k is the specified segment size. Profile 
            groups should be a list of lists, where each sublist contains a number 
            of 2-tuples, the first and second element of which are boath binary
            tuples, specifying the mutation profile of the two segments over 
            which to compute the mean expression shift.

    Returns:
        (np.ndarray): Shape (ngroups,L,L) or (L,L). If a single profile group is 
            specified, either as a single tuple or a single list of tuples, the 
            returned array is the pairwise mean expression shift corresponding 
            to that profile. If multiple profile groups are given, each element 
            of the returned array corresponds to a profile group.
        (list[list[tuple[tuple]]] | list[tuple[tuple]]): Shape (ngroups,?,2,k) 
            or (?,2,K). List of mutation profile groups (each itself a list), if 
            multiple groups are given. If a single profile group is specified, 
            then a single list, specifying the profile. Each group is defined by 
            the list of mutation profile pairs, specified as a 2-tuple of binary 
            k-tuples.
    """
    # Get total expression and counts for each pairwise mutation profile.
    tot_exp_by_idx, counts_by_idx = compute_total_expression_by_pairwise_mutation(
        sequences, expression, wt_seq, segment_size
    )
    nidxs, _, seq_length, _ = tot_exp_by_idx.shape
    
    # Compute the mean expression level for the wildtype.
    mean_wildtype_expression = tot_exp_by_idx[0,0] / counts_by_idx[0,0]

    # Handle mutation profiles.
    if profile_groups is None:
        # If not specified, compute the shift for each mutation profile.
        nprofile_groups = nidxs * nidxs
        profile_groups = []
        for i in range(nidxs):
            profile_i = int_to_binary_arr(i, segment_size)
            for j in range(nidxs):
                profile_j = int_to_binary_arr(j, segment_size)
                profile_groups.append([(profile_i, profile_j)])
        nesting_depth = 4
    else:
        # Otherwise, check that the given group of profiles is consistent.
        nesting_depth = get_nested_depth(profile_groups)
        if nesting_depth == 4:
            nprofile_groups = len(profile_groups)
        elif nesting_depth == 3:
            nprofile_groups = 1
            profile_groups = [profile_groups]
        elif nesting_depth == 2:
            nprofile_groups = 1
            profile_groups = [[profile_groups]]
            assert len(profile_groups)
        else:
            msg = f"Cannot handle given profile groups: {profile_groups}"
            raise RuntimeError(msg)
        profile_groups = [np.array(x) for x in profile_groups]

    total_expression = np.zeros([nprofile_groups, seq_length, seq_length])
    counts = np.zeros([nprofile_groups, seq_length, seq_length])
    for group_idx, profile_group in enumerate(profile_groups):
        for profile1, profile2 in profile_group:
            idx1 = binary_arr_to_int(profile1)  # binary mutation profile
            idx2 = binary_arr_to_int(profile2)  
            total_expression[group_idx] += tot_exp_by_idx[idx1, idx2]
            counts[group_idx] += counts_by_idx[idx1, idx2]
    
    # Compute the average expression
    mean_expression = total_expression / counts
    expression_shift = mean_expression - mean_wildtype_expression
    
    if nesting_depth < 4:
        return expression_shift[0], profile_groups[0]
    else:
        return expression_shift, profile_groups


#####################
##  Miscellaneous  ##
#####################

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
