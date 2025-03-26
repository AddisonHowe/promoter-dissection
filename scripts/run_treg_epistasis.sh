#!/bin/bash
#=============================================================================
#
# FILE: run_treg_epistasis.sh
#
# USAGE: run_treg_epistasis.sh [datdir] [gene] [fname]
#
# DESCRIPTION: Run the treg_epistasis.py script.
#
# EXAMPLE: sh run_treg_epistasis.sh data/treg/lacZ_v1 lacZ_v1 repact_20k
#=============================================================================

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <datdir> <gene> <fname>"
    exit 1
fi

datdir=$1
gene=$2
fname=$3

OUTDIR=results/treg_epistasis
NBOOT=100


python scripts/treg_epistasis.py \
    --datdir ${datdir} \
    --infile ${fname}.csv \
    --sep '\s+' \
    --outdir ${OUTDIR}/${gene}/${fname} \
    --nboot ${NBOOT} \
    --pbar
