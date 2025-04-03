"""Plot results of benchmarking

python tests/benchmarking/plot_results.py

"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('-d', '--datdir', type=str, 
                    default="tests/_tmp")
parser.add_argument('-o', '--outdir', type=str, 
                    default="tests/_tmp")
args = parser.parse_args()

DATDIR = args.datdir
OUTDIR = args.outdir

os.makedirs(OUTDIR, exist_ok=True)

# Load data from benchmark tests

def read_avg_and_std_time(fpath):
    with open(fpath, 'r') as f:
        lines = f.readlines()
        avg_time_lines = [line for line in lines if "avg time:" in line]
        std_time_lines = [line for line in lines if "std time:" in line]
        assert len(avg_time_lines) == 1, "Got more than one matching line."
        assert len(std_time_lines) == 1, "Got more than one matching line."
    avg_time_line = avg_time_lines[0]
    avg_time = float(avg_time_line.removeprefix("avg time: "))
    std_time_line = std_time_lines[0]
    std_time = float(std_time_line.removeprefix("std time: "))
    return avg_time, std_time


###########################
##  Pointwise Functions  ##
###########################

BENCHMARKING_FUNCTION_NAMES = [
    'compute_total_expression_by_mutation',
    'compute_mean_expression_by_mutation',
    'compute_expression_shift_by_mutation',
]

avg_times_reg = np.nan * np.ones(len(BENCHMARKING_FUNCTION_NAMES))
avg_times_jax = np.nan * np.ones(len(BENCHMARKING_FUNCTION_NAMES))
std_times_reg = np.nan * np.ones(len(BENCHMARKING_FUNCTION_NAMES))
std_times_jax = np.nan * np.ones(len(BENCHMARKING_FUNCTION_NAMES))
labels = []
for i, func_name in enumerate(BENCHMARKING_FUNCTION_NAMES):
    fpath_reg = f"{DATDIR}/benchmarking/benchmarks_{func_name}_run1.txt"
    fpath_jax = f"{DATDIR}/jax_benchmarking/benchmarks_{func_name}_run1.txt"
    avg_time_reg, std_time_reg = read_avg_and_std_time(fpath_reg)
    avg_time_jax, std_time_jax = read_avg_and_std_time(fpath_jax)
    avg_times_reg[i] = avg_time_reg
    avg_times_jax[i] = avg_time_jax
    std_times_reg[i] = std_time_reg
    std_times_jax[i] = std_time_jax
    labels.append(
        func_name.removeprefix("compute_").removesuffix("_by_mutation")
    )

# Plot results
fig, ax = plt.subplots(1, 1)
bar_width = 0.35
x_positions = np.arange(len(BENCHMARKING_FUNCTION_NAMES))
bars_reg = ax.bar(
    x_positions - bar_width/2, 
    avg_times_reg, 
    bar_width, 
    label='Standard'
)
bars_jax = ax.bar(
    x_positions + bar_width/2, 
    avg_times_jax, 
    bar_width, 
    label='JAX'
)
# Add annotations
for bar in bars_reg:
    ax.text(
        bar.get_x() + bar.get_width() / 2, bar.get_height(), 
        f'{bar.get_height():.3g}', 
        ha='center', va='bottom', fontsize=10
    )
for bar in bars_jax:
    ax.text(
        bar.get_x() + bar.get_width() / 2, bar.get_height(), 
        f'{bar.get_height():.3g}', 
        ha='center', va='bottom', fontsize=10
    )
ax.set_xlabel('Function')
ax.set_ylabel('Average Time')
ax.set_title('Pointwise function performance')
ax.set_xticks(x_positions)
ax.set_xticklabels(labels, rotation=45, ha='right')
ax.legend()
plt.savefig(f"{OUTDIR}/benchmark_pointwise_jax_comparison.pdf", bbox_inches='tight')
plt.close()


##########################
##  Pairwise Functions  ##
##########################

BENCHMARKING_FUNCTION_NAMES = [
    'compute_total_expression_by_pairwise_mutation',
    'compute_mean_expression_by_pairwise_mutation',
    'compute_expression_shift_by_pairwise_mutation',
]

avg_times_reg = np.nan * np.ones(len(BENCHMARKING_FUNCTION_NAMES))
avg_times_jax = np.nan * np.ones(len(BENCHMARKING_FUNCTION_NAMES))
std_times_reg = np.nan * np.ones(len(BENCHMARKING_FUNCTION_NAMES))
std_times_jax = np.nan * np.ones(len(BENCHMARKING_FUNCTION_NAMES))
labels = []
for i, func_name in enumerate(BENCHMARKING_FUNCTION_NAMES):
    fpath_reg = f"{DATDIR}/benchmarking/benchmarks_{func_name}_run1.txt"
    fpath_jax = f"{DATDIR}/jax_benchmarking/benchmarks_{func_name}_run1.txt"
    avg_time_reg, std_time_reg = read_avg_and_std_time(fpath_reg)
    avg_time_jax, std_time_jax = read_avg_and_std_time(fpath_jax)
    avg_times_reg[i] = avg_time_reg
    avg_times_jax[i] = avg_time_jax
    std_times_reg[i] = std_time_reg
    std_times_jax[i] = std_time_jax
    labels.append(
        func_name.removeprefix("compute_").removesuffix("_by_pairwise_mutation")
    )

# Plot results
fig, ax = plt.subplots(1, 1)
bar_width = 0.35
x_positions = np.arange(len(BENCHMARKING_FUNCTION_NAMES))
bars_reg = ax.bar(
    x_positions - bar_width/2, 
    avg_times_reg, 
    bar_width, 
    label='Standard'
)
bars_jax = ax.bar(
    x_positions + bar_width/2, 
    avg_times_jax, 
    bar_width, 
    label='JAX'
)
# Add annotations
for bar in bars_reg:
    ax.text(
        bar.get_x() + bar.get_width() / 2, bar.get_height(), 
        f'{bar.get_height():.2g}', 
        ha='center', va='bottom', fontsize=10
    )
for bar in bars_jax:
    ax.text(
        bar.get_x() + bar.get_width() / 2, bar.get_height(), 
        f'{bar.get_height():.2g}', 
        ha='center', va='bottom', fontsize=10
    )
ax.set_xlabel('Function')
ax.set_ylabel('Average Time')
ax.set_title('Pairwise function performance')
ax.set_xticks(x_positions)
ax.set_xticklabels(labels, rotation=45, ha='right')
ax.legend()
plt.savefig(f"{OUTDIR}/benchmark_pairwise_jax_comparison.pdf", bbox_inches='tight')
plt.close()




print("Done!")
