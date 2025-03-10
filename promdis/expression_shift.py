"""Expression shift script

Generates plots of expression shift data from an input data file.

Example:

python promdis/expression_shift.py \
    -i data/expression_data/ykgE_dataset_combined.csv -o results/ykgE -g ykgE

"""

import sys
import argparse
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

from promdis.processing import get_sequence_arrays_and_counts, gene_seq_to_array
from promdis.core import compute_pairwise_segmented_mean_expression
from promdis.helpers import binary_arr_to_int
from promdis.pl import plot_data, plot_data_2d


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--infpath', type=str, required=True,
                        help="Input filepath.")
    parser.add_argument('-o', '--outdir', type=str, required=True,
                        help="Output directory.")
    parser.add_argument('-p', '--prefix', type=str, default=None,
                        help="Name prefix for saved plots.")
    parser.add_argument('-g', '--gene', type=str, required=True, 
                        help="Gene name.")
    parser.add_argument('-wt', '--wildtype_fpath', type=str, 
                        default="data/wtsequences.csv", 
                        help="Wildtype gene sequence csv file.")
    parser.add_argument('-s', '--segment_size', type=int, default=2, 
                        help="Segment size.")
    parser.add_argument('--sep', type=str, default=None, 
                        help="Input file column separator.")
    return parser.parse_args(args)


def main(args):
    fpath = args.infpath
    outdir = args.outdir
    gene = args.gene
    wt_genes_fpath = args.wildtype_fpath
    segment_size = args.segment_size
    sep = args.sep
    prefix = args.prefix if args.prefix else gene

    os.makedirs(outdir, exist_ok=True)

    # Load and prepare data
    df = pd.read_csv(fpath, sep=sep)
    df['barcode'] = df['seq'].str.slice(160,)
    df['promoter'] = df['seq'].str.slice(0, 160)

    seqs, counts_dna, counts_rna = get_sequence_arrays_and_counts(
        df, key_promoter='promoter', key_rna='ct_1', key_dna='ct_0'
    )

    expression = counts_rna / (counts_dna + counts_rna)

    # Load gene of interest information
    wildtype_genes_df = pd.read_csv(wt_genes_fpath)
    wt_gene_sequences = {
        g: wildtype_genes_df.loc[wildtype_genes_df['name'] == g,'geneseq'].values[0]
        for g in wildtype_genes_df['name'].unique()
    }
    gene_wt_seq = wt_gene_sequences[gene]
    nbases = len(gene_wt_seq)
    wt_seq = gene_seq_to_array(gene_wt_seq)

    # Compute mu
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mu = compute_pairwise_segmented_mean_expression(
            seqs, expression, wt_seq, 
            segment_size=segment_size
        )

    # Plot mean wildtype expression
    fig, ax = plt.subplots(1, 1)
    sc = ax.imshow(mu[0,0])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = fig.colorbar(sc, cax=cax)
    ax.set_title(f"{gene} mean wildtype expression $\\mu_{{i, j}}^*$")
    ax.set_xlabel(f"segment $i$")
    ax.set_ylabel(f"segment $j$");

    
    saveas = f"{outdir}/{prefix}_mean_wt_expression.pdf"
    plt.savefig(saveas)
    plt.close()

    # Plot xi for various mutation profiles
    mutation_profiles = [
        [(0, 0), (0, 1)],
        [(0, 0), (1, 0)],
        [(0, 0), (1, 1)],
        [(0, 1), (0, 1)],
        [(0, 1), (1, 0)],
        [(0, 1), (1, 1)],
        [(1, 1), (1, 1)],
    ]

    for b1, b2 in mutation_profiles:
        idx1 = binary_arr_to_int(np.array(b1))
        idx2 = binary_arr_to_int(np.array(b2))
        xi = mu[idx1, idx2] - mu[0, 0]
        ax = plot_data_2d(xi)
        ax.set_title(f"Expression shift $\\xi[i,j;{b1},{b2}]$")
        s1 = "".join([str(x) for x in b1])
        s2 = "".join([str(x) for x in b2])
        saveas = f"{outdir}/{prefix}_xi_{s1}_{s2}.pdf"
        plt.savefig(saveas)
        plt.close()
    
    return

if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    main(args)
