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

from promdis.processing import get_sequence_arrays_and_counts, gene_seq_to_array
from promdis.core import compute_mean_expression_by_mutation
from promdis.core import compute_mean_expression_by_pairwise_mutation
from promdis.core import compute_expression_shift_by_mutation
from promdis.core import compute_expression_shift_by_pairwise_mutation
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
    wt_seq = gene_seq_to_array(gene_wt_seq)

    # Compute mean expression in 1 dimension
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mu_1d = compute_mean_expression_by_mutation(
            seqs, expression, wt_seq,
            segment_size=segment_size
        )

    # Plot mean wildtype expression (1 dimension)
    ax = plot_data(
        mu_1d[0], segment_size=1, bin_size=1,
        cmap='viridis',
    )
    ax.set_title(f"{gene} mean wildtype expression $\\mu_{{i}}^*$")
    ax.set_xlabel(f"segment $i$")
    ax.set_ylabel(f"$\\mu$");

    saveas = f"{outdir}/{prefix}_mean_wt_expression_1d.pdf"
    plt.savefig(saveas)
    plt.close()

    # Compute mean expression in 2 dimensions
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mu_2d = compute_mean_expression_by_pairwise_mutation(
            seqs, expression, wt_seq,
            segment_size=segment_size
        )

    # Plot mean wildtype expression (2 dimensions)
    ax = plot_data_2d(
        mu_2d[0,0], norm=None, cmap='viridis',
    )
    ax.set_title(f"{gene} mean wildtype expression $\\mu_{{i,j}}^*$")
    ax.set_xlabel(f"segment $i$")
    ax.set_ylabel(f"segment $j$");

    saveas = f"{outdir}/{prefix}_mean_wt_expression_2d.pdf"
    plt.savefig(saveas)
    plt.close()

    # Plot xi for various mutation profiles (1 dimension)
    mutation_profiles_1d = [
        [(0,1),(1,0),],
        [(1,1)],
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        # Compute expression shift in 1 dimension
        xi_1d_list, profile_groups = compute_expression_shift_by_mutation(
            seqs, expression, wt_seq, 
            segment_size=segment_size,
            profile_groups=mutation_profiles_1d,
        )

        # Plot each group
        for xi, profile in zip(xi_1d_list, profile_groups):
            ax = plot_data(
                xi, segment_size=1, bin_size=1,
            )
            s = ",".join(
                [''.join([str(x) for x in np.array(t).flatten()]) 
                 for t in profile]
            )
            ax.set_title(f"{gene} expression shift $\\xi_{{i}}[{{{s}}}]$")
            ax.set_xlabel(f"segment $i$")
            ax.set_ylabel(f"$\\xi$");
            saveas = f"{outdir}/{prefix}_xi_1d_{s.replace(',', '_')}.pdf"
            plt.savefig(saveas)
            plt.close()


    # Plot xi for various mutation profiles (2 dimensions)
    mutation_profiles_2d = [
        [((0,0),(0,1)), ((0,0),(1,0)), ((0,1),(0,0)), ((1,0),(0,0))],
        [((0,1),(0,1)), ((0,1),(1,0)), ((1,0),(1,0))],
        [((0,0),(1,1)), ((1,1),(0,0))],
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        # Compute expression shift in 2 dimensions
        xi_2d_list, profile_groups = compute_expression_shift_by_pairwise_mutation(
            seqs, expression, wt_seq, 
            segment_size=segment_size,
            profile_groups=mutation_profiles_2d,
        )

        # Plot each group
        for xi, profile in zip(xi_2d_list, profile_groups):
            ax = plot_data_2d(
                xi,
            )
            s = ",".join(
                [''.join([str(x) for x in np.array(t).flatten()]) 
                 for t in profile]
            )
            ax.set_title(f"{gene} expression shift $\\xi_{{i,j}}[{{{s}}}]$")
            ax.set_xlabel(f"segment $i$")
            ax.set_ylabel(f"segment $j$");
            saveas = f"{outdir}/{prefix}_xi_2d_{s.replace(',','_')}.pdf"
            plt.savefig(saveas)
            plt.close()


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    main(args)
