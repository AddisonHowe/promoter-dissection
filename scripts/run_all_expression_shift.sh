#!/bin/bash
#=============================================================================
#
# FILE: run_all_expression_shift.sh
#
# USAGE: run_all_expression_shift.sh [datdir] [gene]
#
# DESCRIPTION: Run the expression_shift.py script on all data files in the 
#  directory <datdir> that correspond to the given gene.
#
# EXAMPLE: sh run_all_expression_shift.sh data/regseq_data ykgE
#=============================================================================

datdir=$1
gene=$2

function process_fpath() {
    fpath=$1
    fname=$(basename $f)
    fname=${fname/_alldone_with_large/}
    echo $fname
}

for f in ${datdir}/${gene}*; do
    fname=$(process_fpath $f)
    echo $fname
    python promdis/expression_shift.py -i $f -o results/$gene/$fname \
        -g $gene -p $fname \
        --sep '\s+'
done
