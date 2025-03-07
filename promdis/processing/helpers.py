"""Data processing helper functions

"""

import math
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

NT_MAP = {c: i for i, c in enumerate(['A', 'C', 'G', 'T'])}


def gene_seq_to_array(seq, mapping=NT_MAP):
    """Convert a string sequence to a numy array, using the provided mapping."""
    if mapping is None:
        mapping = NT_MAP
    return np.array([mapping[c] for c in seq], dtype=np.uint8)


def get_sequence_arrays_and_counts(
        df, *, 
        key_promoter='promoter', 
        key_dna='ct_0', 
        key_rna='ct_1',
        nt_map=NT_MAP,
):
    """Convert Dataframe columns promoter, ct_0, and ct_1 to numpy arrays."""
    promoters = np.array(
        [gene_seq_to_array(s, nt_map) for s in df[key_promoter].values],
    )
    counts_dna = np.array(df[key_dna], dtype=int)
    counts_rna = np.array(df[key_rna], dtype=int)

    return promoters, counts_dna, counts_rna


def compare_sequences(seq1, seq2):
    """Screen for all differing positions between seq1 and seq2.
    Supports broadcasting.
    """
    seq1 = np.broadcast_to(seq1, np.broadcast_shapes(seq1.shape, seq2.shape))
    seq2 = np.broadcast_to(seq2, np.broadcast_shapes(seq1.shape, seq2.shape))
    return seq1 != seq2


def count_mutations(seqs, wt_seq):
    mut_screen = compare_sequences(seqs, wt_seq)
    nmuts = np.sum(mut_screen, axis=-1)
    return nmuts