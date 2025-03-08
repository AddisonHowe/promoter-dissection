"""Tests for data processing functions.

"""

import pytest
import numpy as np
import pandas as pd

from promdis.processing import gene_seq_to_array
from promdis.processing import get_sequence_arrays_and_counts


@pytest.mark.parametrize("seq, mapping, expected", [
    ["ACGT", None, np.array([0, 1, 2, 3])],
    ["ACGTACGT", None, np.array([0, 1, 2, 3, 0, 1, 2, 3])],
    ["TTTT", None, np.array([3, 3, 3, 3])],
])
class TestGeneSeqToArray:
    
    def test_gene_seq_to_array_value(self, seq, mapping, expected):
        result = gene_seq_to_array(seq, mapping=mapping)
        assert np.all(result == expected), f"Got {result}"

    def test_gene_seq_to_array_type(self, seq, mapping, expected):
        result = gene_seq_to_array(seq, mapping=mapping)
        assert isinstance(result[0], np.uint8), f"Got type {type(result[0])}"


@pytest.mark.parametrize("fpath, exp_proms, exp_dna, exp_rna", [
    ['tests/data/test_dataset1.csv', 
     [[0,0,0,0], [1,1,1,1], [2,2,2,2], [3,3,3,3], [0,1,2,3],],
     [1, 1, 1, 2, 2], [0, 1, 2, 4, 3]
    ],
])
def test_get_sequence_arrays_and_counts(fpath, exp_proms, exp_dna, exp_rna):
    df = pd.read_csv(fpath)
    promoters, counts_dna, counts_rna = get_sequence_arrays_and_counts(
        df, key_promoter='seq', key_dna='ct_0', key_rna='ct_1',
    )
    errors = []
    if not np.all(counts_dna == exp_dna):
        msg = f"Wrong DNA counts"
        errors.append(msg)
    if not np.all(counts_rna == exp_rna):
        msg = f"Wrong RNA counts"
        errors.append(msg)
    if not np.all(promoters == exp_proms):
        msg = f"Wrong promoter reads"
        errors.append(msg)
    assert not errors, "Errors occurred:\n{}".format("\n".join(errors))
