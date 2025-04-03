#!/bin/bash
#=============================================================================
#
# FILE: run_treg_epistasis.sh
#
# USAGE: run_treg_epistasis.sh [datdir] [gene] [fname] [optional:seed]
#
# DESCRIPTION: Run the treg_epistasis.py script.
#
# EXAMPLE: sh run_treg_epistasis.sh data/treg/lacZ_v1 lacZ_v1 repact_20k
#=============================================================================

if [ "$#" -eq 3 ]; then
    datdir=$1
    gene=$2
    fname=$3
    seedline=""
elif [ "$#" -eq 4 ]; then
    datdir=$1
    gene=$2
    fname=$3
    seedline="--seed "$4
else
    echo "Usage: $0 <datdir> <gene> <fname> [seed]"
    exit 1
fi



OUTDIR=results/treg_epistasis
NBOOT=100


python scripts/treg_epistasis.py \
    --datdir ${datdir} \
    --infile ${fname}.csv \
    --sep '\s+' \
    --outdir ${OUTDIR}/${gene}/${fname} \
    --nboot ${NBOOT} \
    ${seedline} \
    # --pbar \
