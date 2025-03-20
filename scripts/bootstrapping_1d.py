"""Bootstrapping (1-dimensional)

"""

import argparse
import os
import time
import tqdm as tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import jax
jax.config.update("jax_enable_x64", False)  # TODO: use float64
import jax.numpy as jnp
import jax.random as jrandom
import equinox as eqx

from promdis.processing import get_sequence_arrays_and_counts, gene_seq_to_array
from promdis.jax.core import compute_expression_shift_by_mutation
from promdis.jax.core import compute_expression_shift_by_pairwise_mutation
from promdis.pl import plot_data

#######################
##  Parse arguments  ##
#######################

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--infile', type=str, required=True)
parser.add_argument('-g', '--gene', type=str, required=True)
parser.add_argument('-n', '--nboot', type=int, default=20)
parser.add_argument('--wt_path', type=str, default="data/wtsequences.csv")
parser.add_argument('-o', '--outdir', type=str, default="out/bootstrapping_1d")
parser.add_argument('--seed', type=int, default=None)
parser.add_argument('--sep', type=str, default=None, 
                        help="Input file column separator.")
args = parser.parse_args()

FPATH = args.infile  # "data/expression_data/ykgE_dataset_combined.csv"
GENE = args.gene  # "ykgE"
N_BOOT = args.nboot
WT_GENES_FPATH = args.wt_path
OUTDIR = args.outdir
SEED = args.seed
sep = args.sep

ALPHA = 0.05

IMGDIR = f"{args.outdir}/images"
os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(IMGDIR, exist_ok=True)
rng = np.random.default_rng(seed=SEED)
key = jrandom.PRNGKey(seed=rng.integers(2**32))


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


##################################################
##  Compute expression shift for observed data  ##
##################################################

# All instances with one mutation at a single position. 
xi_data, _ = compute_expression_shift_by_mutation(
    seqs, expression, wt_seq, 
    segment_size=1,
    profile_groups=[(1,)]
)


# Plotting 
ax = plot_data(
    xi_data,
    segment_size=1,
    cmap='RdBu',
)
ax.set_title("$\\xi_i$")
plt.savefig(f"{IMGDIR}/xi_data.pdf")
plt.close()


#############################
##  Perform bootstrapping  ##
#############################

def generate_bootstrapped_data(
        seqs, expression, wt_seq, key
):
    """Generate random wildtype mutants."""
    key, subkey = jrandom.split(key, 2)
    idxs = jrandom.choice(subkey, jnp.arange(len(seqs)), shape=(len(seqs),), replace=True)
    boot_seqs = seqs[idxs,:]
    boot_expression = expression[idxs]
    return boot_seqs, boot_expression


@eqx.filter_jit
def bootstrap_computations(seqs, expression, wt_seq, key):
    boot_seqs, boot_expression = generate_bootstrapped_data(
        seqs, expression, wt_seq, key, 
    )

    # All instances with one mutation at a single position. 
    xi_1mut, _ = compute_expression_shift_by_mutation(
        boot_seqs, boot_expression, wt_seq, 
        segment_size=1,
        profile_groups=[(1,)]
    )

    return xi_1mut


print("Generating bootstrap data...")
xis_boot = jnp.zeros([N_BOOT, *xi_data.shape])
time0 = time.time()
for i in tqdm.trange(N_BOOT):
    key, subkey = jrandom.split(key, 2)
    boot_xi_1mut = bootstrap_computations(
        seqs, expression, wt_seq, subkey
    )
    xis_boot = xis_boot.at[i].set(boot_xi_1mut)

time1 = time.time()
print(f"Completed bootstrapping in {time1-time0:.3f} sec.")

print("xis_boot contains nan?: ", np.any(np.isnan(xis_boot)))

# Compute bootstrapped confidence intervals
xi_boot_ci_lower = np.nanpercentile(xis_boot, 100*ALPHA/2, axis=0)
xi_boot_ci_upper = np.nanpercentile(xis_boot, 100*(1-ALPHA/2), axis=0)
xi_boot_median = np.nanpercentile(xis_boot, 50, axis=0)
np.save(f"{OUTDIR}/xi_boot_ci_lower.npy", xi_boot_ci_lower)
np.save(f"{OUTDIR}/xi_boot_ci_upper.npy", xi_boot_ci_upper)
np.save(f"{OUTDIR}/xi_boot_median.npy", xi_boot_median)


################
##  Plotting  ##
################

ax = plot_data(
    xi_boot_median,
    segment_size=1,
)
ax.set_title(f"Boostrapped $\\xi_{{i}}$")
plt.savefig(f"{IMGDIR}/bootstrapped_xi_median.pdf")
plt.close()

# Plot only regions where confident
conf_screen = xi_boot_ci_lower * xi_boot_ci_upper > 0
ax = plot_data(
    np.where(conf_screen, xi_boot_median, 0.),
    segment_size=1,
)
ax.set_title(f"Boostrapped $\\xi_{{i}}$")
plt.savefig(f"{IMGDIR}/bootstrapped_xi_confident.pdf")
plt.close()

# Main plot
fig, ax = plt.subplots(1, 1)
xs = np.arange(1, len(xi_data) + 1)

for i in range(N_BOOT):
    xi = xis_boot[i]
    colors = ['b' if v < 0 else 'r' for v in xi]
    ax.scatter(xs, xi, color=colors, s=1, alpha=0.1, rasterized=True)

ax.scatter(xs, xi_boot_ci_lower, color='k', alpha=0.5, s=5, marker='_')
ax.scatter(xs, xi_boot_ci_upper, color='k', alpha=0.5, s=5, marker='_')

conf_screen = xi_boot_ci_lower * xi_boot_ci_upper > 0

colors = []
for i in range(len(conf_screen)):
    if conf_screen[i]:
        v = xi_data[i]
        colors.append('cyan' if v < 0 else 'purple')
    else:
        colors.append('grey')

ax.scatter(xs, xi_data, color=colors, alpha=1, s=10)

ax.set_title("Bootstrapped $\\xi_i$");
plt.savefig(f"{IMGDIR}/bootstrapped_xi.pdf")
plt.close()
