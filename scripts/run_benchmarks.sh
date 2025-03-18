#!/bin/bash
#=============================================================================
#
# FILE: run_benchmarks.sh
#
# USAGE: run_benchmarks.sh
#
# DESCRIPTION: Run benchmarks for standard and JAX accelerated functions, then
#  plot the results of the benchmark tests.
#
# EXAMPLE: sh run_benchmarks.sh
#=============================================================================

eval "$(conda shell.bash hook)"
conda activate env


pytest -s --benchmark tests/benchmarking/benchmark_core.py
pytest -s --benchmark tests/benchmarking/benchmark_jax_core.py

python tests/benchmarking/plot_results.py
