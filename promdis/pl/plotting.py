"""Plotting functions

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable


def plot_data(
        values, 
        segment_size=1, 
        bin_size=1,
        seq_start=None,
        **kwargs,
):
    """Convenience plotting function for binned, colored bar plots."""
    fig, ax = plt.subplots(1, 1, figsize=kwargs.get('figsize', None))
    
    color = kwargs.get('color', None)

    # Bin data
    num_bins = len(values) // bin_size
    binned_vals = [
        np.mean(values[i*bin_size:(i+1)*bin_size]) for i in range(num_bins)
    ]

    # Generate x values for binned data
    x = np.arange(num_bins) * bin_size * segment_size

    if color is None:
        colors = ['red' if value > 0 else 'blue' for value in binned_vals]
    else:
        colors = [color] * len(binned_vals)
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


def plot_data_2d(
        values,
        nan_color='k',
        norm=True,
        vmax=None,
        **kwargs,
):
    """Convenience plotting function for 2D segmented expression data."""
    fig, ax = plt.subplots(1, 1, figsize=kwargs.get('figsize', None))
    
    cmap = kwargs.get('cmap', plt.cm.bwr.copy())
    if isinstance(cmap, str):
        cmap = plt.cm.get_cmap(cmap)
    if nan_color:
        cmap.set_bad(color=nan_color)

    if norm:
        vmax = np.nanmax(np.abs(values)) if vmax is None else vmax
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    sc = ax.imshow(values.T, cmap=cmap, norm=norm)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = fig.colorbar(sc, cax=cax)
    ax.set_xlabel(kwargs.get('xlabel', "segment $i$"))
    ax.set_ylabel(kwargs.get('ylabel', "segment $j$"))

    return ax
