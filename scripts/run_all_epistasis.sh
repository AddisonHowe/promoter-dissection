#!/bin/bash
#=============================================================================
#
# FILE: run_all_epistasis.sh
#
# USAGE: run_all_epistasis.sh [datdir] [gene]
#
# DESCRIPTION: Run the epistasis.py script on all data files in the 
#  directory <datdir> that correspond to the given gene.
#
# EXAMPLE: sh run_all_epistasis.sh data/regseq_data ykgE
#=============================================================================

datdir=$1
gene=$2

OUTDIR=results/epistasis_by_gene_and_cond
NBOOT=100

function process_fpath() {
    fpath=$1
    fname=$(basename $f)
    fname=${fname/_alldone_with_large/}
    echo $fname
}

for f in ${datdir}/${gene}*; do
    fname=$(process_fpath $f)
    echo $fname
    python scripts/epistasis.py \
        --infile $f \
        --sep '\s+' \
        --gene $gene \
        --outdir ${OUTDIR}/$gene/$fname \
        --nboot ${NBOOT}
done
