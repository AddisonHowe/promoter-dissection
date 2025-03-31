"""Theoretical RegSeq Epistasis

"""

import argparse
import os
import time
import tqdm as tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import jax.random as jrandom
import equinox as eqx

from promdis.processing import get_sequence_arrays_and_counts, gene_seq_to_array
from promdis.jax.core import compute_expression_shift_by_mutation
from promdis.jax.core import compute_expression_shift_by_pairwise_mutation
from promdis.jax.core import compute_epistasis_effect
from promdis.pl import plot_data_2d, DEFAULT_CMAP

CMAP = DEFAULT_CMAP

#######################
##  Parse arguments  ##
#######################

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--datdir', type=str, required=True)
parser.add_argument('-i', '--infile', type=str, required=True)
parser.add_argument('-n', '--nboot', type=int, default=20)
parser.add_argument('-o', '--outdir', type=str, default="out/treg_epistasis")
parser.add_argument('--seed', type=int, default=None)
parser.add_argument('--pbar', action='store_true')
parser.add_argument('--float64', action='store_true')
parser.add_argument('--sep', type=str, default=None, 
                        help="Input file column separator.")
parser.add_argument('--img_format', type=str, choices=['pdf', 'png'], 
                    default='pdf', help="Format for output images.")
args = parser.parse_args()

datdir = args.datdir  # "data/treg/lacZ_v1"
FNAME = args.infile  # constitutive_20k.csv
N_BOOT = args.nboot
OUTDIR = args.outdir
SEED = args.seed
sep = args.sep
disable_pbar = not args.pbar
use_float64 = args.float64
img_format = args.img_format

if use_float64:
    print("Using dtype float64")
    jax.config.update("jax_enable_x64", True)
    
ALPHA = 0.05

IMGDIR = f"{args.outdir}/images"
os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(IMGDIR, exist_ok=True)

if SEED is None:
    SEED = np.random.randint(1, 10**12)
print(f"Seed: {SEED}")
np.savetxt(f"{OUTDIR}/seed.txt", [SEED], fmt='%d')
rng = np.random.default_rng(seed=SEED)
key = jrandom.PRNGKey(seed=rng.integers(2**32))


###############################################
##  Load data and compute expression values  ##
###############################################

df = pd.read_csv(f"{datdir}/{FNAME}")
df['barcode'] = df['seq'].str.slice(160,)
df['promoter'] = df['seq'].str.slice(0, 160)
df
seqs, counts_dna, counts_rna = get_sequence_arrays_and_counts(
    df, key_promoter='promoter', key_rna='ct_1', key_dna='ct_0'
)
expression = counts_rna / (counts_dna + counts_rna)

# Load wildtype data
gene = np.genfromtxt(f"{datdir}/gene_name.txt", dtype=str).item()
gene_wt_seq = np.genfromtxt(f"{datdir}/wtseq.txt", dtype=str).item()
nbases = len(gene_wt_seq)
wt_seq = gene_seq_to_array(gene_wt_seq)


##################################################
##  Compute epistasis effect for observed data  ##
##################################################

eta, (xi_1mut, xi_2mut) = compute_epistasis_effect(
    seqs, expression, wt_seq, 
    segment_size=1,
)

# Save results
np.save(f"{OUTDIR}/xi_1mut.npy", xi_1mut)
np.save(f"{OUTDIR}/xi_2mut.npy", xi_2mut)
np.save(f"{OUTDIR}/eta.npy", eta)

# Plotting 
ax = plot_data_2d(
    xi_2mut,
    cmap=CMAP,
)
ax.set_title("$\\xi[i,j]$ for 2 mutations across 2 segments")
plt.savefig(f"{IMGDIR}/xi_pairwise.{img_format}")
plt.close()

ax = plot_data_2d(
    eta,
    cmap=CMAP,
)
ax.set_title("$\\eta[i,j]$")
plt.savefig(f"{IMGDIR}/eta.{img_format}")
plt.close()

ax = plot_data_2d(
    eta,
    cmap=CMAP,
    vmax=0.3,
)
ax.set_title("$\\eta[i,j]$")
plt.savefig(f"{IMGDIR}/eta_vmax_03.{img_format}")
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

    # Compute eta
    eta, (xi_1mut, xi_2mut) = compute_epistasis_effect(
        boot_seqs, boot_expression, wt_seq, 
        segment_size=1,
    )
    return eta, xi_1mut, xi_2mut


print("Generating bootstrap data...")
xis_boot = jnp.zeros([N_BOOT, *xi_2mut.shape])
etas_boot = jnp.zeros([N_BOOT, *eta.shape])
time0 = time.time()
for i in tqdm.trange(N_BOOT, disable=disable_pbar):
    key, subkey = jrandom.split(key, 2)
    boot_eta, boot_xi_1mut, boot_xi_2mut = bootstrap_computations(
        seqs, expression, wt_seq, subkey
    )
    xis_boot = xis_boot.at[i].set(boot_xi_2mut)
    etas_boot = etas_boot.at[i].set(boot_eta)

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

eta_boot_ci_lower = np.nanpercentile(etas_boot, 100*ALPHA/2, axis=0)
eta_boot_ci_upper = np.nanpercentile(etas_boot, 100*(1-ALPHA/2), axis=0)
eta_boot_median = np.nanpercentile(etas_boot, 50, axis=0)
np.save(f"{OUTDIR}/eta_boot_ci_lower.npy", eta_boot_ci_lower)
np.save(f"{OUTDIR}/eta_boot_ci_upper.npy", eta_boot_ci_upper)
np.save(f"{OUTDIR}/eta_boot_median.npy", eta_boot_median)


################
##  Plotting  ##
################

ax = plot_data_2d(
    xi_boot_median
)
ax.set_title(f"Boostrapped $\\xi_{{i,j}}$")
plt.savefig(f"{IMGDIR}/bootstrapped_xi_median.{img_format}")
plt.close()


# Plot only regions where confident
conf_screen = xi_boot_ci_lower * xi_boot_ci_upper > 0
ax = plot_data_2d(
    np.where(conf_screen, xi_boot_median, 0.),
)
ax.set_title(f"Boostrapped $\\xi_{{i,j}}$")
plt.savefig(f"{IMGDIR}/bootstrapped_xi_confident.{img_format}")
plt.close()


ax = plot_data_2d(
    eta_boot_median
)
ax.set_title(f"Boostrapped $\\eta_{{i,j}}$")
plt.savefig(f"{IMGDIR}/bootstrapped_eta_median.{img_format}")
plt.close()


# Plot only regions where confident
conf_screen = eta_boot_ci_lower * eta_boot_ci_upper > 0
ax = plot_data_2d(
    np.where(conf_screen, eta_boot_median, 0.),
)
ax.set_title(f"Boostrapped $\\eta_{{i,j}}$")
plt.savefig(f"{IMGDIR}/bootstrapped_eta_confident.{img_format}")
plt.close()
