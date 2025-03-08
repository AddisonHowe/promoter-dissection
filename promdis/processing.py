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
    """Convert Dataframe columns `promoter`, `ct_0`, and `ct_1` to numpy arrays.

    Args:
        df (Dataframe): Dataframe with columns corresponding to the promoter
            sequence, DNA counts, and RNA counts.
        key_promoter (str, optional): Promoter column name. Defaults to 'promoter'.
        key_dna (str, optional): DNA counts column name. Defaults to 'ct_0'.
        key_rna (str, optional): RNA counts column name. Defaults to 'ct_1'.
        nt_map (dict, optional): Dictionary mapping sequence character values to
            np.uint8 values. Defaults to NT_MAP={'A':0, 'C':1, 'G':2, 'T':3}.

    Returns:
        np.ndarray[np.uint8]: Array of sequences. Shape (num_seqs, seq_length).
        np.ndarray[int]: Array of DNA counts. Shape (num_seqs,)
        np.ndarray[int]: Array of RNA counts. Shape (num_seqs,)
    """    
    promoters = np.array(
        [gene_seq_to_array(s, nt_map) for s in df[key_promoter].values],
    )
    counts_dna = np.array(df[key_dna], dtype=int)
    counts_rna = np.array(df[key_rna], dtype=int)

    return promoters, counts_dna, counts_rna
