#!/bin/bash
#=============================================================================
#
# FILE: run_basic_plots.sh
#
# USAGE: run_basic_plots.sh [datdir] [gene]
#
# DESCRIPTION: Run the basic_plots.py script on all data files in the 
#  directory <datdir> that correspond to the given gene.
#
# EXAMPLE: sh run_basic_plots.sh data/regseq_data ykgE
#=============================================================================

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <datdir> <gene>"
    exit 1
fi

datdir=$1
gene=$2

OUTDIR=results/basic_plots_by_gene_and_cond

function process_fpath() {
    fpath=$1
    fname=$(basename $f)
    fname=${fname/_alldone_with_large/}
    echo $fname
}

for f in ${datdir}/${gene}*; do
    fname=$(process_fpath $f)
    echo $fname
    python scripts/basic_plots.py \
        --infile $f \
        --sep '\s+' \
        --gene $gene \
        --outdir ${OUTDIR}/$gene/$fname
done
