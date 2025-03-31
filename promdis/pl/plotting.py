"""Plotting functions

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

DEFAULT_CMAP = "RdBu_r"

def plot_data(
        values, *,
        segment_size=1, 
        bin_size=1,
        tss=115,
        tick_start=0,
        tick_spacing=20,
        color=None,
        ax=None,
        **kwargs,
):
    """Convenience plotting function for binned, colored bar plots.
    
    Args:
        (np.ndarray) values: Data to plot.
        (int) segment_size: Length of each sequence segment. Default 1.
        (int) bin_size: Binning parameter for smoothing. Default 1.
        (int) tss: Location of the transcription start site, which will be 
            labeled as position 0. Default 115.
        (int) tick_start: Starting location for tickmarks, relative to the tss.
            Default 0, so that the tss has a tick mark. Ticks will extend above
            and below the start, to the extent of the image.
        (int) tick_spacing: Interval between ticks. Default 20.
        (str) color: If specified, the color for all bars. Default None.
        (plt.Axes | None) ax: Axis object. If None, creates a new plt.Axes.

    Keyword Args:
        figsize
        xlabel
        ylabel
    
    Returns:
        plt.Axis: Axis object.
    """

    figsize=kwargs.get('figsize', None)
    xlabel = kwargs.get('xlabel', 'position')
    ylabel = kwargs.get('ylabel', None)

    if np.ndim(values) != 1:
        msg = f"Got {np.ndim(values)}-dimensional input values. Should be 1D."
        raise RuntimeError(msg)
    
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.get_figure()
    
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
    ax.bar(
        x, binned_vals, 
        width=bin_width, 
        color=colors, 
        align='edge'
    )

    # Set the x-axis labels according to the given sequence interval
    # if seq_start is not None:
    #     xticks = ax.get_xticks()
    #     new_labels = xticks.astype(int) + seq_start
    #     ax.xaxis.set_major_locator(plt.FixedLocator(xticks))
    #     ax.set_xticklabels(new_labels)
    
    # Modify ticks
    x_min, x_max = ax.get_xlim()
    x_ticks = np.concatenate([
        np.arange(tss + tick_start, x_min, -tick_spacing, dtype=int), 
        np.arange(tss + tick_start, x_max, tick_spacing, dtype=int)
    ])
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([str(t - tss) for t in x_ticks])
    ax.set_xlim(x_min, x_max)

    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    return ax


def plot_data_2d(
        values, *,
        nan_color='k',
        norm=True,
        vmax=None,
        tss=115,
        tick_start=0,
        tick_spacing=20,
        cmap=None,
        ax=None,
        **kwargs,
):
    """Convenience plotting function for 2D segmented expression data.
    
    Args:
        (np.ndarray) values: Data to plot. The first row, data[0], is plotted
            as the first column.
        (str) nan_color: Color assigned to nan values. Default 'k'.
        (bool) norm: Whether to normalize colors between -vmax and vmax, where 
            vmax is the largest data magnitude, or specified as an argument.
        (None | float) vmax: Maximum value on the coloring scale, if specified.
            If None, determined based on the given data values. Only utilized 
            when norm=True. Default None.
        (int) tss: Location of the transcription start site, which will be 
            labeled as position 0. Default 115.
        (int) tick_start: Starting location for tickmarks, relative to the tss.
            Default 0, so that the tss has a tick mark. Ticks will extend above
            and below the start, to the extent of the image.
        (int) tick_spacing: Interval between ticks. Default 20.
        (str | plt.colormap | None) cmap: Colormap. Default None. Either a 
            string specifying a valid plt.Colormap, or a plt.Colormap. If None,
            defaults to the DEFAULT_CMAP defined in pl.plotting.py.
        (plt.Axes | None) ax: Axis object. If None, creates a new plt.Axes.

    Keyword Args:
        figsize
        xlabel
        ylabel
    
    Returns:
        plt.Axis: Axis object.
    """

    figsize = kwargs.get('figsize', None)
    xlabel = kwargs.get('xlabel', "position $i$")
    ylabel = kwargs.get('ylabel', "position $j$")
    
    if np.ndim(values) != 2:
        msg = f"Got {np.ndim(values)}-dimensional input values. Should be 2D."
        raise RuntimeError(msg)
    
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.get_figure()
    
    if cmap is None:
        cmap = plt.cm.get_cmap(DEFAULT_CMAP).copy()
    elif isinstance(cmap, str):
        cmap = plt.cm.get_cmap(cmap)
    elif isinstance(cmap, plt.Colormap):
        pass
    else:
        msg = f"Cannot handle cmap argument of type {type(cmap)}"
        raise RuntimeError(msg)
    
    if nan_color:
        cmap.set_bad(color=nan_color)

    if norm:
        vmax = np.nanmax(np.abs(values)) if vmax is None else vmax
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    
    sc = ax.imshow(
        values.T, cmap=cmap, norm=norm, 
        origin='lower',
        interpolation='none',
    )

    # Modify ticks
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    x_ticks = np.concatenate([
        np.arange(tss + tick_start, x_min, -tick_spacing, dtype=int), 
        np.arange(tss + tick_start, x_max, tick_spacing, dtype=int)
    ])
    y_ticks = np.concatenate([
        np.arange(tss + tick_start, y_min, -tick_spacing, dtype=int),
        np.arange(tss + tick_start, y_max, tick_spacing, dtype=int)
    ])
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.set_xticklabels([str(t - tss) for t in x_ticks])
    ax.set_yticklabels([str(t - tss) for t in y_ticks])
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = fig.colorbar(sc, cax=cax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    return ax
