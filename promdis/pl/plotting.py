"""Plotting functions

"""

import numpy as np
import matplotlib.pyplot as plt


def plot_data(
        values, 
        segment_size=1, 
        bin_size=1,
        seq_start=None,
        **kwargs,
):
    """Convenience plotting function for binned, colored bar plots."""
    fig, ax = plt.subplots(1, 1, figsize=kwargs.get('figsize', None))
    
    # Bin data
    num_bins = len(values) // bin_size
    binned_vals = [
        np.mean(values[i*bin_size:(i+1)*bin_size]) for i in range(num_bins)
    ]

    # Generate x values for binned data
    x = np.arange(num_bins) * bin_size * segment_size

    colors = ['red' if value < 0 else 'blue' for value in binned_vals]
    bin_width = bin_size * segment_size
    ax.bar(x, binned_vals, width=bin_width, color=colors, align='edge')

    # Set the x-axis labels according to the given sequence interval
    if seq_start is not None:
        xticks = ax.get_xticks()
        new_labels = xticks.astype(int) + seq_start
        ax.xaxis.set_major_locator(plt.FixedLocator(xticks))
        ax.set_xticklabels(new_labels)
    
    ax.set_xlabel('position')
    ax.set_ylabel(kwargs.get('ylabel', None))

    return ax
