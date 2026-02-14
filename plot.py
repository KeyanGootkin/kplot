import warnings
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, Colormap
CmapTypes = [ListedColormap, LinearSegmentedColormap, Colormap]

from kplot.Axes import access_subplots
from kplot.utils import alias_kwarg, column_width, parse_multiax_params

def decode_plot_src(src) -> tuple:
    match src:
        case (np.ndarray()|list()|tuple(), np.ndarray()|list()|tuple()):
            return np.array(src[0]), np.array(src[1])
        case _: raise TypeError(f"plotting function recieved source of type {type(src)}, which is not supported.")

def _plot_one_line(plot_dict, line_dict, axes_dict, **kwargs):
    x, y, ax = plot_dict['x'], plot_dict['y'], plot_dict['axes']
    color, lw, ls, m, ms = line_dict['color'], line_dict['linewidth'], line_dict['linestyle'], line_dict['marker'], line_dict['markersize']
    artist = ax.plot(
        x, y, 
        color = color if type(color)==str else color(0.01),
        lw=lw, ls=ls, marker=m, ms=ms,
        **kwargs
    )
    # set scales 
    ax.set_yscale(axes_dict['yscale'])
    ax.set_xscale(axes_dict['xscale'])
    # set limits
    ax.set_xlim(axes_dict['xlim'])
    ax.set_ylim(axes_dict['ylim'])
    # set labels
    ax.set_xlabel(axes_dict['xlabel'])
    ax.set_ylabel(axes_dict['ylabel'])
    # set title
    ax.set_title(axes_dict['title'])
    # set aspect
    ax.set_aspect(axes_dict['aspect'])
    return artist

def _plot_multiple_lines(plot_dict: dict, line_dict: dict, axes_dict: dict, N: int, multiax: bool, **kwargs):
    fig = plot_dict['fig']
    axes = parse_multiax_params(plot_dict['axes'], [Axes], N)
    xs = parse_multiax_params(plot_dict['x'], [np.ndarray], N, out_ndim=1)
    ys = parse_multiax_params(plot_dict['y'], [np.ndarray], N, out_ndim=1)

    match line_dict['color']:
        case str()|tuple(): 
            colors = parse_multiax_params(line_dict['color'], [str, tuple], N)
        case x if type(line_dict['color']) in CmapTypes: 
            colors = [line_dict['color'](i/(N+1)) for i in range(N)]
        case [int(), *_] | [float(), *_]: 
            mn, mx = np.nanmin(line_dict['color']), np.nanmax(line_dict['color'])
            colors = [line_dict['cmap']((x + mn)/(mx-mn)) for x in line_dict['color']]
    lws = parse_multiax_params(line_dict['linewidth'], [float, int], N) 
    lss = parse_multiax_params(line_dict['linestyle'], [str], N)
    ms = parse_multiax_params(line_dict['marker'], [str], N)
    mss = parse_multiax_params(line_dict['markersize'], [float], N)

    xlims = parse_multiax_params(axes_dict['xlim'], [tuple], N)
    ylims = parse_multiax_params(axes_dict['ylim'], [tuple], N)
    xlabels = parse_multiax_params(axes_dict['xlabel'], [str], N)
    ylabels = parse_multiax_params(axes_dict['ylabel'], [str], N)
    xscale = parse_multiax_params(axes_dict['xscale'], [str], N)
    yscale = parse_multiax_params(axes_dict['yscale'], [str], N)
    titles = parse_multiax_params(axes_dict['title'], [str], N)
    aspects = parse_multiax_params(axes_dict['aspect'], [str], N)
    lines = []
    for i in range(len(ys)):
        ax = axes[i]
        lines.append(ax.plot(xs[i], ys[i], color=colors[i], ls=lss[i], lw=lws[i], marker=ms[i], ms=mss[i]))
    return fig, axes, lines

def plot(
    *src,
    #figure setup
    fig=None, 
    axes=None, ax=None,
    figsize: tuple = (column_width, column_width), fs: tuple = None,
    show: bool = False,
    close: bool = False,
    #line formating
    color: str|Colormap|list[int] = "black",
    cmap: str|Colormap = plt.cm.plasma,
    linewidth: int = None, lw: int = None,
    linestyle: str = None, ls: str = None,
    #marker formatting
    marker: str = None, 
    markersize: int = None, ms: int = None,
    #plot formating
    xlabel: None|str = None,
    ylabel: None|str = None, 
    xlim: tuple = (None, None),
    ylim: tuple = (None, None),
    scale: str = None,
    xscale: str = None, 
    yscale: str = None,
    title: str|None = None,
    aspect: str = 'auto',
    #presentation parameters
    save: str = "",
    dpi: int = 100,
    #everything else goes into plot
    **kwargs
):
    # decode args
    x, y = decode_plot_src(src)
    # parse aliased keywords
    axes = alias_kwarg("axes", axes, "ax", ax)
    figsize = fs if fs else figsize
    linewidth = alias_kwarg("linewidth", linewidth, "lw", lw)
    linestyle = alias_kwarg("linestyle", linestyle, "ls", ls)
    markersize = alias_kwarg("markersize", markersize, 'ms', ms)
    if scale: xscale = yscale = scale 
    # prep figure
    fig, axes = access_subplots(fig=fig, axes=axes, figsize=figsize)
    plot_dict = {
        'x': x,
        'y': y,
        'fig': fig,
        'axes': axes
    }
    line_dict = {
        'color': color,
        'cmap': cmap,
        'linewidth': linewidth,
        'linestyle': linestyle,
        'marker': marker,
        'markersize': markersize
    }
    axes_dict = {
        'xlim': xlim, 
        'ylim': ylim,
        'xlabel': xlabel,
        'ylabel': ylabel,
        'xscale': xscale if xscale else "linear",
        'yscale': yscale if yscale else "linear",
        'title': title,
        'aspect': aspect
    }

    # interpret x and y based on the number of dimensions they have
    match x.ndim, y.ndim:
        case 1, 1:
            assert x.shape==y.shape, f"x and y must be same shape, \n\tx: {x.shape}\n\t{y.shape}"
            plot_dict['axes'] = axes[0]
            lines = _plot_one_line(plot_dict, line_dict, axes_dict, **kwargs)
            axes = axes[0]
        case 1, 2:
            assert all(len(x)==len(yi) for yi in y), fr"all y arrays must be the same length as the x array\n\tx: {len(x)}\n\ty: {"\n\t   ".join([f"{i}: {len(yi)}" for i,yi in enumerate(y)])}"
            multiax = len(y)==len(axes)
            fig, axes, lines = _plot_multiple_lines(plot_dict, line_dict, axes_dict, len(y), multiax)
            axes = axes if multiax else axes[0]
        case 2, 2:
            assert len(x)==len(y), f"x and y must have the same number of lines\n\tx: {len(x)}\n\ty: {len(y)}"
            multiax = len(y)==len(axes)
            lines = [_plot_one_line(xi, yi, fig, ax, **kwargs) for xi, yi, ax in zip(x, y, axes)] if multiax else [_plot_one_line(xi, yi, fig, axes[0]) for xi, yi in zip(x, y)]
            axes = axes if multiax else axes[0]
    # present the figure
    if len(save)>0: 
        if not '.' in save: save +=".jpg"
        plt.savefig(save, dpi=dpi)
    if show: plt.show()
    if close: plt.close(fig)
    return fig, axes, lines   

def _align_algorithm(x: list|np.ndarray, mode: str):
    match mode:
        case 'mid'|'m'|'center'|'c':
            return [(x[i] + x[i+1])/2 for i in range(len(x)-1)]
        case 'logmid'|'lm':
            return [np.log10((10**x[i] + 10**x[i+1])/ 2) for i in range(len(x)-1)]
        case 'left'|'l':
            return x[:-1]
        case 'right'|'r':
            return x[1:]

def diffplot(x, y, align: str = 'mid', **kwargs):
    dx = np.diff(x)
    dy = np.diff(y)
    x_aligned = _align_algorithm(x, align)
    return plot(x_aligned, dy/dx, **kwargs)

def powerlawplot(x, y, align: str = 'logmid', **kwargs):
    return diffplot(np.log10(x), np.log10(y), align=align, **kwargs)

def colored_line(x, y, c, ax, **lc_kwargs):
    """
    Plot a line with a color specified along the line by a third value.

    It does this by creating a collection of line segments. Each line segment is
    made up of two straight lines each connecting the current (x, y) point to the
    midpoints of the lines connecting the current point with its two neighbors.
    This creates a smooth line with no gaps between the line segments.

    Parameters
    ----------
    x, y : array-like
        The horizontal and vertical coordinates of the data points.
    c : array-like
        The color values, which should be the same size as x and y.
    ax : Axes
        Axis object on which to plot the colored line.
    **lc_kwargs
        Any additional arguments to pass to matplotlib.collections.LineCollection
        constructor. This should not include the array keyword argument because
        that is set to the color argument. If provided, it will be overridden.

    Returns
    -------
    matplotlib.collections.LineCollection
        The generated line collection representing the colored line.
    """
    if "array" in lc_kwargs:
        warnings.warn('The provided "array" keyword argument will be overridden')

    # Default the capstyle to butt so that the line segments smoothly line up
    default_kwargs = {"capstyle": "butt"}
    default_kwargs.update(lc_kwargs)

    # Compute the midpoints of the line segments. Include the first and last points
    # twice so we don't need any special syntax later to handle them.
    x = np.asarray(x)
    y = np.asarray(y)
    x_midpts = np.hstack((x[0], 0.5 * (x[1:] + x[:-1]), x[-1]))
    y_midpts = np.hstack((y[0], 0.5 * (y[1:] + y[:-1]), y[-1]))

    # Determine the start, middle, and end coordinate pair of each line segment.
    # Use the reshape to add an extra dimension so each pair of points is in its
    # own list. Then concatenate them to create:
    # [
    #   [(x1_start, y1_start), (x1_mid, y1_mid), (x1_end, y1_end)],
    #   [(x2_start, y2_start), (x2_mid, y2_mid), (x2_end, y2_end)],
    #   ...
    # ]
    coord_start = np.column_stack((x_midpts[:-1], y_midpts[:-1]))[:, np.newaxis, :]
    coord_mid = np.column_stack((x, y))[:, np.newaxis, :]
    coord_end = np.column_stack((x_midpts[1:], y_midpts[1:]))[:, np.newaxis, :]
    segments = np.concatenate((coord_start, coord_mid, coord_end), axis=1)

    lc = LineCollection(segments, **default_kwargs)
    lc.set_array(c)  # set the colors of each segment

    return ax.add_collection(lc)
