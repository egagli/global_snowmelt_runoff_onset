import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize


def create_2d_hist_with_1d_marginals(pandas_df, x_variable, y_variable,
                                      c_variable, reduce_C_function,
                                      bins=30, mincnt=1000, cmap='inferno',
                                      alpha=0.8,
                                      vmin=None, vmax=None,
                                      cbar_label='Value',
                                      x_label=None, y_label=None,
                                      x_min=None, x_max=None,
                                      y_min=None, y_max=None,
                                      x_bin_width=None, y_bin_width=None,
                                      plot_type='hexbin',
                                      figsize=(10, 8)):
    """
    Create a 2D histogram with 1D marginal histograms where the marginal bars
    are colored according to the reduced C values in each bin.

    Parameters:
    -----------
    pandas_df : pd.DataFrame
        DataFrame containing the data
    x_variable : str
        Column name for x-axis variable
    y_variable : str
        Column name for y-axis variable
    c_variable : str
        Column name for the variable to reduce (color by)
    reduce_C_function : callable
        Function to reduce C values in each bin (e.g., np.mean, np.median)
    bins : int or array-like
        Number of bins or bin edges for histograms
    mincnt : int
        Minimum count per bin to display
    cmap : str
        Colormap name
    vmin, vmax : float
        Color scale limits
    cbar_label : str
        Label for colorbar
    x_label, y_label : str
        Axis labels
    x_min, x_max : float
        X-axis limits for binning and plotting
    y_min, y_max : float
        Y-axis limits for binning and plotting
    x_bin_width, y_bin_width : float
        Bin widths for x and y axes (overrides bins if provided)
    plot_type : str
        Type of 2D plot: 'hexbin' or '2dhist'
    figsize : tuple
        Figure size (width, height)

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object
    """

    # Extract data
    x_data = pandas_df[x_variable].values
    y_data = pandas_df[y_variable].values
    c_data = pandas_df[c_variable].values

    # Remove NaN values
    mask = ~(np.isnan(x_data) | np.isnan(y_data) | np.isnan(c_data))
    x_clean = x_data[mask]
    y_clean = y_data[mask]
    c_clean = c_data[mask]

    # Set up the figure with gridspec
    fig = plt.figure(figsize=figsize, dpi=300)
    gs = GridSpec(4, 3, figure=fig,
                  width_ratios=[3, 0.3, 0.1],
                  height_ratios=[0.3, 3, 0.5, 0.2],
                  hspace=0.02, wspace=0.02)

    # Main 2D histogram
    ax_main = fig.add_subplot(gs[1, 0])

    # Marginal histogram axes
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)

    # Colorbar axis (positioned below with proper spacing)
    ax_cbar = fig.add_subplot(gs[3, 0])

    # Create 2D histogram
    if plot_type == 'hexbin':
        hb = ax_main.hexbin(x_clean, y_clean, C=c_clean,
                            reduce_C_function=reduce_C_function,
                            gridsize=bins, cmap=cmap, mincnt=mincnt,
                            vmin=vmin, vmax=vmax, alpha=alpha)
    elif plot_type == '2dhist':
        # For 2D histogram, we need to calculate the bin edges
        if x_bin_width is not None and y_bin_width is not None:
            x_min_use = x_min if x_min is not None else x_clean.min()
            x_max_use = x_max if x_max is not None else x_clean.max()
            y_min_use = y_min if y_min is not None else y_clean.min()
            y_max_use = y_max if y_max is not None else y_clean.max()

            x_bins_2d = np.arange(x_min_use, x_max_use + x_bin_width,
                                  x_bin_width)
            y_bins_2d = np.arange(y_min_use, y_max_use + y_bin_width,
                                  y_bin_width)
        elif isinstance(bins, int):
            x_min_use = x_min if x_min is not None else x_clean.min()
            x_max_use = x_max if x_max is not None else x_clean.max()
            y_min_use = y_min if y_min is not None else y_clean.min()
            y_max_use = y_max if y_max is not None else y_clean.max()

            x_bins_2d = np.linspace(x_min_use, x_max_use, bins + 1)
            y_bins_2d = np.linspace(y_min_use, y_max_use, bins + 1)
        else:
            x_bins_2d = bins
            y_bins_2d = bins

        # Calculate the 2D histogram with C values
        hist_2d, x_edges_2d, y_edges_2d = np.histogram2d(
            x_clean, y_clean, bins=[x_bins_2d, y_bins_2d])

        # Calculate reduced C values for each bin
        c_values_2d = np.full(hist_2d.shape, np.nan)
        for i in range(len(x_edges_2d) - 1):
            for j in range(len(y_edges_2d) - 1):
                mask_2d = ((x_clean >= x_edges_2d[i]) &
                           (x_clean < x_edges_2d[i + 1]) &
                           (y_clean >= y_edges_2d[j]) &
                           (y_clean < y_edges_2d[j + 1]))
                if np.sum(mask_2d) >= mincnt:
                    c_values_2d[i, j] = reduce_C_function(c_clean[mask_2d])

        # Plot as pcolormesh - transpose c_values_2d to match expectations
        # pcolormesh expects C to be (M-1, N-1) where X,Y are (M, N)
        hb = ax_main.pcolormesh(x_edges_2d, y_edges_2d, c_values_2d.T,
                                cmap=cmap, vmin=vmin, vmax=vmax, alpha=alpha)
    else:
        raise ValueError("plot_type must be 'hexbin' or '2dhist'")

    # Get the bin edges and centers for marginal histograms
    if x_bin_width is not None and y_bin_width is not None:
        x_min_use = x_min if x_min is not None else x_clean.min()
        x_max_use = x_max if x_max is not None else x_clean.max()
        y_min_use = y_min if y_min is not None else y_clean.min()
        y_max_use = y_max if y_max is not None else y_clean.max()

        x_bins = np.arange(x_min_use, x_max_use + x_bin_width, x_bin_width)
        y_bins = np.arange(y_min_use, y_max_use + y_bin_width, y_bin_width)
    elif isinstance(bins, int):
        x_min_use = x_min if x_min is not None else x_clean.min()
        x_max_use = x_max if x_max is not None else x_clean.max()
        y_min_use = y_min if y_min is not None else y_clean.min()
        y_max_use = y_max if y_max is not None else y_clean.max()

        x_bins = np.linspace(x_min_use, x_max_use, bins + 1)
        y_bins = np.linspace(y_min_use, y_max_use, bins + 1)
    else:
        x_bins = bins
        y_bins = bins

    # Calculate marginal histograms with C values
    # X marginal (top)
    x_hist_counts, x_bin_edges = np.histogram(x_clean, bins=x_bins)
    x_bin_centers = (x_bin_edges[:-1] + x_bin_edges[1:]) / 2

    # Calculate reduced C values for each x bin
    x_c_values = []
    for i in range(len(x_bin_edges) - 1):
        mask_x = ((x_clean >= x_bin_edges[i]) &
                  (x_clean < x_bin_edges[i + 1]))
        if np.sum(mask_x) >= mincnt:
            x_c_values.append(reduce_C_function(c_clean[mask_x]))
        else:
            x_c_values.append(np.nan)

    # Y marginal (right)
    y_hist_counts, y_bin_edges = np.histogram(y_clean, bins=y_bins)
    y_bin_centers = (y_bin_edges[:-1] + y_bin_edges[1:]) / 2

    # Calculate reduced C values for each y bin
    y_c_values = []
    for i in range(len(y_bin_edges) - 1):
        mask_y = ((y_clean >= y_bin_edges[i]) &
                  (y_clean < y_bin_edges[i + 1]))
        if np.sum(mask_y) >= mincnt:
            y_c_values.append(reduce_C_function(c_clean[mask_y]))
        else:
            y_c_values.append(np.nan)

    # Normalize colors based on the same scale as main plot
    norm = Normalize(vmin=vmin if vmin is not None else np.nanmin(c_clean),
                     vmax=vmax if vmax is not None else np.nanmax(c_clean))

    # Plot marginal histograms with colors
    # X marginal
    valid_x = ~np.isnan(x_c_values)
    if np.any(valid_x):
        ax_top.bar(x_bin_centers[valid_x], x_hist_counts[valid_x],
                   width=np.diff(x_bin_edges)[0] * 0.95,
                   color=plt.cm.get_cmap(cmap)(
                       norm(np.array(x_c_values)[valid_x])),
                   alpha=alpha, edgecolor='black', linewidth=1)

    # Y marginal
    valid_y = ~np.isnan(y_c_values)
    if np.any(valid_y):
        ax_right.barh(y_bin_centers[valid_y], y_hist_counts[valid_y],
                      height=np.diff(y_bin_edges)[0] * 0.95,
                      color=plt.cm.get_cmap(cmap)(
                          norm(np.array(y_c_values)[valid_y])),
                      alpha=alpha, edgecolor='black', linewidth=1)

    # Format axes
    ax_main.grid(True, alpha=0.6)
    ax_main.set_xlabel(x_label if x_label else x_variable)
    ax_main.set_ylabel(y_label if y_label else y_variable)

    # draw dashed lines 2d hist bins
    for x_edge in x_bin_edges:
        ax_main.axvline(x_edge, color='gray', linestyle='--', linewidth=0.5, alpha=0.3)
    for y_edge in y_bin_edges:
        ax_main.axhline(y_edge, color='gray', linestyle='--', linewidth=0.5, alpha=0.3)

    # Set axis limits if provided
    if x_min is not None and x_max is not None:
        ax_main.set_xlim(x_min, x_max)
    if y_min is not None and y_max is not None:
        ax_main.set_ylim(y_min, y_max)

    # Clean up marginal plots - remove all labels, ticks, and spines
    ax_top.tick_params(labelbottom=False, labeltop=False,
                       labelleft=False, labelright=False,
                       bottom=False, top=False, left=False, right=False)
    ax_right.tick_params(labelbottom=False, labeltop=False,
                         labelleft=False, labelright=False,
                         bottom=False, top=False, left=False, right=False)

    # Remove spines (borders) from marginal plots
    for spine in ax_top.spines.values():
        spine.set_visible(False)
    for spine in ax_right.spines.values():
        spine.set_visible(False)

    # Add horizontal colorbar with proper spacing
    cbar = plt.colorbar(hb, cax=ax_cbar, orientation='horizontal')
    cbar.set_label(cbar_label)

    return fig


def median_absolute_deviation(x):
    """Calculate median absolute deviation"""
    if len(x) == 0:
        return np.nan
    return np.median(np.abs(x - np.median(x)))
