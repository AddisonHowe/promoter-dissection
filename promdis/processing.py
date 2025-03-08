"""Data processing functions

"""

import numpy as np
import pandas as pd


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
