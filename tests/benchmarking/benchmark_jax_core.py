"""Benchmarking Core functions with JAX acceleration

pytest -s --benchmark tests/benchmarking/benchmark_jax_core.py

"""

import pytest
import os
import numpy as np
import pandas as pd
import timeit
import jax
import jax.numpy as jnp

import equinox as eqx

from promdis.processing import get_sequence_arrays_and_counts, gene_seq_to_array

from promdis.jax.core import compute_total_expression_by_mutation
from promdis.jax.core import compute_mean_expression_by_mutation
from promdis.jax.core import compute_expression_shift_by_mutation
from promdis.jax.core import compute_total_expression_by_pairwise_mutation
from promdis.jax.core import compute_mean_expression_by_pairwise_mutation
from promdis.jax.core import compute_expression_shift_by_pairwise_mutation
from promdis.jax.helpers import get_segments


#####################
##  Configuration  ##
#####################

DATDIR = f"tests/_tmp"
OUTDIR = f"{DATDIR}/jax_benchmarking"

os.makedirs(OUTDIR, exist_ok=True)

INFPATH = "tests/data/ykgE_dataset_combined.csv"
WTFPATH = "tests/data/wtsequences.csv"

def load_data():
    df = pd.read_csv(INFPATH)
    df['barcode'] = df['seq'].str.slice(160,)
    df['promoter'] = df['seq'].str.slice(0, 160)
    seqs, counts_dna, counts_rna = get_sequence_arrays_and_counts(
        df, key_promoter='promoter', key_rna='ct_1', key_dna='ct_0'
    )
    expression = counts_rna / (counts_dna + counts_rna)
    # Load wildtype data
    wildtype_genes_df = pd.read_csv(WTFPATH)
    gene = 'ykgE'
    wt_gene_sequences = {
        g: wildtype_genes_df.loc[
            wildtype_genes_df['name'] == g,'geneseq'].values[0]
        for g in wildtype_genes_df['name'].unique()
    }
    gene_wt_seq = wt_gene_sequences[gene]
    wt_seq = gene_seq_to_array(gene_wt_seq)
    return seqs, expression, wt_seq



##############################################################################
###########################   BEGIN BENCHMARKING   ###########################
##############################################################################


@pytest.mark.benchmark
@pytest.mark.parametrize("name, segment_size", [
    ['run1', 1],
])
def test_compute_total_expression_by_mutation(name, segment_size):

    NITERS = 20

    func_name = 'compute_total_expression_by_mutation'
    func = eqx.filter_jit(compute_total_expression_by_mutation)
    sequences, expression, wt_seq = load_data()
    
    # Warmup
    time0 = timeit.default_timer()
    result = func(
        sequences, expression, wt_seq, segment_size
    )
    jax.block_until_ready(result)
    time1 = timeit.default_timer()
    total_warmup_time = time1 - time0
    
    # Time
    times = []
    for i in range(NITERS):
        time0 = timeit.default_timer()
        result = func(
            sequences, expression, wt_seq, segment_size,
        )
        time1 = timeit.default_timer()
        times.append(time1 - time0)
    avg_time = np.mean(times)
    std_time = np.std(times)

    log_fpath = f"{OUTDIR}/benchmarks_{func_name}_{name}.txt"
    with open(log_fpath, 'w') as f:
        f.write(f"Benchmark results for function: {func_name}\n")
        f.write(f"--------------------------------------------------------\n")
        f.write(f"total warmup time: {total_warmup_time}\n")
        f.write(f"num iterations: {NITERS}\n")
        f.write(f"avg time: {avg_time}\n")
        f.write(f"std time: {std_time}\n")


@pytest.mark.benchmark
@pytest.mark.parametrize("name, segment_size", [
    ['run1', 1],
])
def test_compute_mean_expression_by_mutation(name, segment_size):

    NITERS = 20

    func_name = 'compute_mean_expression_by_mutation'
    func = eqx.filter_jit(compute_mean_expression_by_mutation)
    sequences, expression, wt_seq = load_data()
    
    # Warmup
    time0 = timeit.default_timer()
    result = func(
        sequences, expression, wt_seq, segment_size
    )
    jax.block_until_ready(result)
    time1 = timeit.default_timer()
    total_warmup_time = time1 - time0
    
    # Time
    times = []
    for i in range(NITERS):
        time0 = timeit.default_timer()
        result = func(
            sequences, expression, wt_seq, segment_size,
        )
        time1 = timeit.default_timer()
        times.append(time1 - time0)
    avg_time = np.mean(times)
    std_time = np.std(times)

    log_fpath = f"{OUTDIR}/benchmarks_{func_name}_{name}.txt"
    with open(log_fpath, 'w') as f:
        f.write(f"Benchmark results for function: {func_name}\n")
        f.write(f"--------------------------------------------------------\n")
        f.write(f"total warmup time: {total_warmup_time}\n")
        f.write(f"num iterations: {NITERS}\n")
        f.write(f"avg time: {avg_time}\n")
        f.write(f"std time: {std_time}\n")


@pytest.mark.benchmark
@pytest.mark.parametrize("name, segment_size, profile_groups", [
    ['run1', 1, None],
])
def test_compute_expression_shift_by_mutation(name, segment_size, profile_groups):

    NITERS = 20

    func_name = 'compute_expression_shift_by_mutation'
    func = eqx.filter_jit(compute_expression_shift_by_mutation)
    sequences, expression, wt_seq = load_data()
    
    # Warmup
    time0 = timeit.default_timer()
    result = func(
        sequences, expression, wt_seq, segment_size, profile_groups
    )
    jax.block_until_ready(result)
    time1 = timeit.default_timer()
    total_warmup_time = time1 - time0
    
    # Time
    times = []
    for i in range(NITERS):
        time0 = timeit.default_timer()
        result = func(
            sequences, expression, wt_seq, segment_size, profile_groups
        )
        time1 = timeit.default_timer()
        times.append(time1 - time0)
    avg_time = np.mean(times)
    std_time = np.std(times)

    log_fpath = f"{OUTDIR}/benchmarks_{func_name}_{name}.txt"
    with open(log_fpath, 'w') as f:
        f.write(f"Benchmark results for function: {func_name}\n")
        f.write(f"--------------------------------------------------------\n")
        f.write(f"total warmup time: {total_warmup_time}\n")
        f.write(f"num iterations: {NITERS}\n")
        f.write(f"avg time: {avg_time}\n")
        f.write(f"std time: {std_time}\n")


###############################
##  Pairwise Core Functions  ##
###############################


@pytest.mark.benchmark
@pytest.mark.parametrize("name, segment_size", [
    ['run1', 1],
])
def test_compute_total_expression_by_pairwise_mutation(name, segment_size):

    NITERS = 20

    func_name = 'compute_total_expression_by_pairwise_mutation'
    func = eqx.filter_jit(compute_total_expression_by_pairwise_mutation)
    sequences, expression, wt_seq = load_data()
    segments = jnp.asarray(get_segments(
        sequences, segment_size, 
        startpos=0, 
        stride=segment_size
    ))
    
    # Warmup
    time0 = timeit.default_timer()
    result = func(
        sequences, expression, wt_seq, segment_size, segments=segments,
    )
    jax.block_until_ready(result)
    time1 = timeit.default_timer()
    total_warmup_time = time1 - time0
    
    # Time
    times = []
    for i in range(NITERS):
        time0 = timeit.default_timer()
        result = func(
            sequences, expression, wt_seq, segment_size, segments=segments,
        )
        time1 = timeit.default_timer()
        times.append(time1 - time0)
    avg_time = np.mean(times)
    std_time = np.std(times)

    log_fpath = f"{OUTDIR}/benchmarks_{func_name}_{name}.txt"
    with open(log_fpath, 'w') as f:
        f.write(f"Benchmark results for function: {func_name}\n")
        f.write(f"--------------------------------------------------------\n")
        f.write(f"total warmup time: {total_warmup_time}\n")
        f.write(f"num iterations: {NITERS}\n")
        f.write(f"avg time: {avg_time}\n")
        f.write(f"std time: {std_time}\n")


@pytest.mark.benchmark
@pytest.mark.parametrize("name, segment_size", [
    ['run1', 1],
])
def test_compute_mean_expression_by_pairwise_mutation(name, segment_size):

    NITERS = 20

    func_name = 'compute_mean_expression_by_pairwise_mutation'
    func = eqx.filter_jit(compute_mean_expression_by_pairwise_mutation)
    sequences, expression, wt_seq = load_data()
    
    # Warmup
    time0 = timeit.default_timer()
    result = func(
        sequences, expression, wt_seq, segment_size
    )
    jax.block_until_ready(result)
    time1 = timeit.default_timer()
    total_warmup_time = time1 - time0
    
    # Time
    times = []
    for i in range(NITERS):
        time0 = timeit.default_timer()
        result = func(
            sequences, expression, wt_seq, segment_size,
        )
        time1 = timeit.default_timer()
        times.append(time1 - time0)
    avg_time = np.mean(times)
    std_time = np.std(times)

    log_fpath = f"{OUTDIR}/benchmarks_{func_name}_{name}.txt"
    with open(log_fpath, 'w') as f:
        f.write(f"Benchmark results for function: {func_name}\n")
        f.write(f"--------------------------------------------------------\n")
        f.write(f"total warmup time: {total_warmup_time}\n")
        f.write(f"num iterations: {NITERS}\n")
        f.write(f"avg time: {avg_time}\n")
        f.write(f"std time: {std_time}\n")


@pytest.mark.benchmark
@pytest.mark.parametrize("name, segment_size, profile_groups", [
    ['run1', 1, None],
])
def test_compute_expression_shift_by_pairwise_mutation(name, segment_size, profile_groups):

    NITERS = 20

    func_name = 'compute_expression_shift_by_pairwise_mutation'
    func = eqx.filter_jit(compute_expression_shift_by_pairwise_mutation)
    sequences, expression, wt_seq = load_data()
    
    # Warmup
    time0 = timeit.default_timer()
    result = func(
        sequences, expression, wt_seq, segment_size, profile_groups
    )
    jax.block_until_ready(result)
    time1 = timeit.default_timer()
    total_warmup_time = time1 - time0
    
    # Time
    times = []
    for i in range(NITERS):
        time0 = timeit.default_timer()
        result = func(
            sequences, expression, wt_seq, segment_size, profile_groups
        )
        time1 = timeit.default_timer()
        times.append(time1 - time0)
    avg_time = np.mean(times)
    std_time = np.std(times)

    log_fpath = f"{OUTDIR}/benchmarks_{func_name}_{name}.txt"
    with open(log_fpath, 'w') as f:
        f.write(f"Benchmark results for function: {func_name}\n")
        f.write(f"--------------------------------------------------------\n")
        f.write(f"total warmup time: {total_warmup_time}\n")
        f.write(f"num iterations: {NITERS}\n")
        f.write(f"avg time: {avg_time}\n")
        f.write(f"std time: {std_time}\n")
