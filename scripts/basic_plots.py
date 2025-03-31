"""Basic plots

"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import jax

from promdis.processing import get_sequence_arrays_and_counts, gene_seq_to_array
from promdis.jax.helpers import int_to_binary_arr
from promdis.jax.core import compute_mean_expression_by_mutation
from promdis.jax.core import compute_mean_expression_by_pairwise_mutation
from promdis.jax.core import compute_expression_shift_by_mutation
from promdis.jax.core import compute_expression_shift_by_pairwise_mutation
from promdis.jax.core import compute_epistasis_effect
from promdis.pl import plot_data, plot_data_2d

#######################
##  Parse arguments  ##
#######################

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--infile', type=str, required=True)
parser.add_argument('-g', '--gene', type=str, required=True)
parser.add_argument('--wt_path', type=str, default="data/wtsequences.csv")
parser.add_argument('-o', '--outdir', type=str, default="out/basic_plots")
parser.add_argument('--float64', action='store_true')
parser.add_argument('--sep', type=str, default=None, 
                        help="Input file column separator.")
parser.add_argument('--img_format', type=str, choices=['pdf', 'png'], 
                    default='pdf', help="Format for output images.")
args = parser.parse_args()

FPATH = args.infile  # "data/expression_data/ykgE_dataset_combined.csv"
GENE = args.gene  # "ykgE"
WT_GENES_FPATH = args.wt_path
OUTDIR = args.outdir
sep = args.sep
use_float64 = args.float64
img_format = args.img_format

if use_float64:
    print("Using dtype float64")
    jax.config.update("jax_enable_x64", True)
    
IMGDIR = f"{args.outdir}/images"
os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(IMGDIR, exist_ok=True)


###############################################
##  Load data and compute expression values  ##
###############################################

df = pd.read_csv(FPATH, sep=sep)
df['barcode'] = df['seq'].str.slice(160,)
df['promoter'] = df['seq'].str.slice(0, 160)
df
seqs, counts_dna, counts_rna = get_sequence_arrays_and_counts(
    df, key_promoter='promoter', key_rna='ct_1', key_dna='ct_0'
)
expression = counts_rna / (counts_dna + counts_rna)

# Load wildtype data
wildtype_genes_df = pd.read_csv(WT_GENES_FPATH)
wt_gene_sequences = {
    g: wildtype_genes_df.loc[wildtype_genes_df['name'] == g,'geneseq'].values[0]
    for g in wildtype_genes_df['name'].unique()
}
gene_wt_seq = wt_gene_sequences[GENE]
nbases = len(gene_wt_seq)
wt_seq = gene_seq_to_array(gene_wt_seq)


###############################################################
##  Compute mean expression levels for wildtype and mutants  ##
###############################################################

#~~~  Singlets  ~~~#
mus_single = compute_mean_expression_by_mutation(
    seqs, expression, wt_seq, segment_size=1,
)

ax = plot_data(
    mus_single[0], segment_size=1, bin_size=1,
    color='green'
)
ax.set_title(f"$\\mu_i^*$ (singlets)")
plt.savefig(f"{IMGDIR}/mu_singlet_0.{img_format}")
plt.close()

ax = plot_data(
    mus_single[1], segment_size=1, bin_size=1,
    color='green'
)
ax.set_title(f"$\\mu_i[1]$ (singlets)")
plt.savefig(f"{IMGDIR}/mu_singlet_1.{img_format}")
plt.close()

#~~~  Doublets  ~~~#
mus_double = compute_mean_expression_by_mutation(
    seqs, expression, wt_seq, segment_size=2,
)
ax = plot_data(
    mus_double[0], segment_size=2, bin_size=1,
    color='green'
)
ax.set_title(f"$\\mu_i^*$ (doublets)")
plt.savefig(f"{IMGDIR}/mu_doublet_00.{img_format}")
plt.close()

ax = plot_data(
    mus_double[1], segment_size=2, bin_size=1,
    color='green'
)
ax.set_title(f"$\\mu_i[01]$ (doublets)")
plt.savefig(f"{IMGDIR}/mu_doublet_01.{img_format}")
plt.close()

ax = plot_data(
    mus_double[2], segment_size=2, bin_size=1,
    color='green'
)
ax.set_title(f"$\\mu_i[10]$ (doublets)")
plt.savefig(f"{IMGDIR}/mu_doublet_10.{img_format}")
plt.close()

ax = plot_data(
    mus_double[3], segment_size=2, bin_size=1,
    color='green'
)
ax.set_title(f"$\\mu_i[11]$ (doublets)")
plt.savefig(f"{IMGDIR}/mu_doublet_11.{img_format}")
plt.close()


########################################################################
##  Compute pairwise mean expression levels for wildtype and mutants  ##
########################################################################

#~~~  Singlets  ~~~#
pairwise_mus_single = compute_mean_expression_by_pairwise_mutation(
    seqs, expression, wt_seq, segment_size=1,
)

ax = plot_data_2d(
    pairwise_mus_single[0][0], cmap='viridis', norm=None,
)
ax.set_title(f"$\\mu_{{i,j}}^{{**}}$ (singlets)")
plt.savefig(f"{IMGDIR}/pairwise_mu_singlet_00.{img_format}")
plt.close()

profiles = [(0,1), (1,0), (1,1)]
for i, j in profiles:
    ax = plot_data_2d(
        pairwise_mus_single[i][j], cmap='viridis', norm=None,
    )
    ax.set_title(f"$\\mu_{{i,j}}[{i},{j}]$")
    plt.savefig(f"{IMGDIR}/pairwise_mu_singlet_{i}{j}.{img_format}")
    plt.close()

#~~~  Doublets  ~~~#
pairwise_mus_double = compute_mean_expression_by_pairwise_mutation(
    seqs, expression, wt_seq, segment_size=2,
)

ax = plot_data_2d(
    pairwise_mus_double[0][0], cmap='viridis', norm=None,
)
ax.set_title(f"$\\mu_{{i,j}}^{{**}}$ (doublets)")
plt.savefig(f"{IMGDIR}/pairwise_mu_doublet_00.{img_format}")
plt.close()

profiles = [
    (0,1),
    (1,0),
    (1,1),
]
for i, j in profiles:
    ax = plot_data_2d(
        pairwise_mus_double[i][j], cmap='viridis', norm=None,
    )
    b1 = "".join(map(str, int_to_binary_arr(i, n=2)))
    b2 = "".join(map(str, int_to_binary_arr(j, n=2)))
    ax.set_title(f"$\\mu_{{i,j}}[{b1},{b2}]$")
    plt.savefig(f"{IMGDIR}/pairwise_mu_doublet_{b1}{b2}.{img_format}")
    plt.close()


#################################################################
##  Compute expression shifts and epistasis effect (singlets)  ##
#################################################################

# All instances with one mutation at a given position. 
xi_singlet_1, _ = compute_expression_shift_by_mutation(
    seqs, expression, wt_seq, 
    segment_size=1,
    profile_groups=[(1,)]
)

# All instances with one mutation on both segments.
pairwise_xi_singlet_11, _ = compute_expression_shift_by_pairwise_mutation(
    seqs, expression, wt_seq,
    segment_size=1,
    profile_groups=[((1,),(1,))],
)

# Difference between two-segment mutations and *two* one-segment mutations
eta_singlet = pairwise_xi_singlet_11 - xi_singlet_1[None,:] - xi_singlet_1[:,None]
print("eta contains nan?: ", np.any(np.isnan(eta_singlet)))

# Plotting 
ax = plot_data(
    xi_singlet_1, segment_size=1, bin_size=1,
    cmap='RdBu_r',
)
ax.set_title("$\\xi_{i}$")
plt.savefig(f"{IMGDIR}/xi_singlet_1.{img_format}")
plt.close()

ax = plot_data_2d(
    pairwise_xi_singlet_11,
    cmap='RdBu_r',
)
ax.set_title("$\\xi_{i,j}$")
plt.savefig(f"{IMGDIR}/pairwise_xi_singlet_11.{img_format}")
plt.close()

ax = plot_data_2d(
    eta_singlet,
    cmap='RdBu_r',
)
ax.set_title("$\\eta_{i,j}$")
plt.savefig(f"{IMGDIR}/eta_singlet.{img_format}")
plt.close()


#################################################################
##  Compute expression shifts and epistasis effect (doublets)  ##
#################################################################

# All instances with exactly one mutation to a segment. 
xi_doublet_01_10, _ = compute_expression_shift_by_mutation(
    seqs, expression, wt_seq, 
    segment_size=2,
    profile_groups=[(0,1),(1,0)]
)

# All instances with exactly one mutation on both segments.
pairwise_xi_doublet_0001_0010, _ = compute_expression_shift_by_pairwise_mutation(
    seqs, expression, wt_seq,
    segment_size=2,
    profile_groups=[((0,0),(0,1)), ((0,0),(1,0))],
)

# All instances with exactly one mutation on both segments.
pairwise_xi_doublet_0101_0110_1001_1010, _ = compute_expression_shift_by_pairwise_mutation(
    seqs, expression, wt_seq,
    segment_size=2,
    profile_groups=[((0,1),(0,1)), ((0,1),(1,0)), ((1,0),(0,1)), ((1,0),(1,0))],
)

# Difference between two-segment mutations and *two* one-segment mutations
eta_doublet_v1 = pairwise_xi_doublet_0101_0110_1001_1010 \
    - xi_doublet_01_10[None,:] \
    - xi_doublet_01_10[:,None]

eta_doublet_v2 = pairwise_xi_doublet_0101_0110_1001_1010 \
    - pairwise_xi_doublet_0001_0010 \
    - pairwise_xi_doublet_0001_0010.T

print("Largest discrepancy between eta_doublet_v1 and eta_doublet_v2:", 
      np.nanmax(np.abs(eta_doublet_v1 - eta_doublet_v2)))

# Plotting 
ax = plot_data(
    xi_doublet_01_10, segment_size=1, bin_size=1,
    cmap='RdBu_r',
)
ax.set_title("$\\xi_{i}[01|10]$")
plt.savefig(f"{IMGDIR}/xi_doublet_01_10.{img_format}")
plt.close()

ax = plot_data_2d(
    pairwise_xi_doublet_0101_0110_1001_1010,
    cmap='RdBu_r',
)
ax.set_title("$\\xi_{i,j}[(01,01)|(01,10)|(10,01)|(10,10)]$")
plt.savefig(f"{IMGDIR}/pairwise_xi_doublet_0101_0110_1001_1010.{img_format}")
plt.close()

ax = plot_data_2d(
    eta_doublet_v1,
    cmap='RdBu_r',
)
ax.set_title("$\\eta_{i,j}$")
plt.savefig(f"{IMGDIR}/eta_v1_doublet.{img_format}")
plt.close()

ax = plot_data_2d(
    eta_doublet_v2,
    cmap='RdBu_r',
)
ax.set_title("$\\eta_{i,j}$")
plt.savefig(f"{IMGDIR}/eta_v2_doublet.{img_format}")
plt.close()


#######################################################################
##  Compute epistasis effect using function (singlets and doublets)  ##
#######################################################################

eta_singlet_official, _ = compute_epistasis_effect(
    seqs, expression, wt_seq, segment_size=1
)

eta_doublet_official, _ = compute_epistasis_effect(
    seqs, expression, wt_seq, segment_size=2
)

ax = plot_data_2d(
    eta_singlet_official,
    cmap='RdBu_r',
)
ax.set_title("$\\eta_{i,j}$")
plt.savefig(f"{IMGDIR}/eta_official_singlet.{img_format}")
plt.close()

ax = plot_data_2d(
    eta_doublet_official,
    cmap='RdBu_r',
)
ax.set_title("$\\eta_{i,j}$")
plt.savefig(f"{IMGDIR}/eta_official_doublet.{img_format}")
plt.close()
