#!/bin/bash
#=============================================================================
#
# FILE: run_all_bootstrapping_1d.sh
#
# USAGE: run_all_bootstrapping_1d.sh [datdir] [gene]
#
# DESCRIPTION: Run the bootstrapping_1d.py script on all data files in the 
#  directory <datdir> that correspond to the given gene.
#
# EXAMPLE: sh run_all_bootstrapping_1d.sh data/regseq_data ykgE
#=============================================================================

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <datdir> <gene>"
    exit 1
fi

datdir=$1
gene=$2

OUTDIR=results/bootstrapping_1d
NBOOT=10000

function process_fpath() {
    fpath=$1
    fname=$(basename $f)
    fname=${fname/_alldone_with_large/}
    echo $fname
}

for f in ${datdir}/${gene}*; do
    fname=$(process_fpath $f)
    echo $fname
    python scripts/bootstrapping_1d.py \
        --infile $f \
        --sep '\s+' \
        --gene $gene \
        --outdir ${OUTDIR}/$gene/$fname \
        --nboot ${NBOOT}
done
