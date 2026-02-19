# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
# >-|===|>                             Imports                             <|===|-<
# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==

from kplot.utils import alias_kwarg, column_width
from kplot.axes import access_subplots
from kplot.utils import alias_kwarg, column_width, parse_multiax_params
from kplot.cmaps import Cmap
import numpy as np
import matplotlib.pyplot as plt

# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
# >-|===|>                           Definitions                           <|===|-<
# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
# >-|===|>                            Functions                            <|===|-<
# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==

def decode_hist_src(src) -> tuple:
    match src:
        case np.ndarray()|list(): return src 
def hist(
    *src,
    #figure setup
    fig=None, 
    axes=None, ax=None,
    figsize: tuple = (column_width, column_width), fs: tuple = None,
    show: bool = False,
    close: bool = False,
    #line formating
    color: str|Cmap|list[int] = "black",
    cmap: str|Cmap = plt.cm.plasma,
    linewidth: int = None, lw: int = None,
    linestyle: str = None, ls: str = None,
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
    **kwargs
):
    # decode args
    xs = decode_hist_src(src)
    # parse aliased keywords
    axes = alias_kwarg("axes", axes, "ax", ax)
    figsize = fs if fs else figsize
    linewidth = alias_kwarg("linewidth", linewidth, "lw", lw)
    linestyle = alias_kwarg("linestyle", linestyle, "ls", ls)
    if scale: xscale = yscale = scale 
    # prep figure
    fig, axes = access_subplots(fig=fig, axes=axes, figsize=figsize)
    plot_dict = {
        'xs': xs,
        'fig': fig,
        'axes': axes
    }
    line_dict = {
        'color': color,
        'cmap': cmap,
        'linewidth': linewidth,
        'linestyle': linestyle,
    }
    axes_dict = {
        'xlim': xlim, 
        'ylim': ylim,
        'xlabel': xlabel,
        'ylabel': ylabel,
        'xscale': xscale if xscale is not None else "linear",
        'yscale': yscale if yscale is not None else "linear",
        'title': title,
        'aspect': aspect
    }
    xs = tuple(a if type(a)!=list else np.array(a) for a in xs)
    match args:
        case (): raise TypeError("kplot.hist missing 1 required positional argument")
        case np.ndarray(),: 
            x = xs[0]
            nx, xbins, h = ax.hist(x)
            histograms = [h]
        case np.ndarray(), np.ndarray():
            x = xs[0]
            y = xs[1]
            nxy, xbins, ybins, h = ax.hist2d(x, y)
            histograms.append(h)

    # present the figure
    if len(save)>0: 
        if not '.' in save: save +=".jpg"
        plt.savefig(save, dpi=dpi)
    if show: plt.show()
    if close: plt.close(fig)
    return fig, axes, histograms  

# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
# >-|===|>                             Classes                             <|===|-<
# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
# >-|===|>                            Decorators                           <|===|-<
# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==