# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
# >-|===|>                             Imports                             <|===|-<
# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
from kplot.axes import subplots

from kbasic.typing import Number, Iterable, Optional
from numpy import asarray, hstack, column_stack, newaxis, concatenate, hypot, diff,\
                nanmin, nanmax, ndarray, log10
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection

# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
# >-|===|>                            Functions                            <|===|-<
# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
def line2segments(
        x: Iterable, y: Iterable
        ) -> ndarray:
    """convert an x, y line into individual [(x1, y1), ..., (xn, yn)] segments with a midpoint. Stolen from stack exchange"""
    # Compute the midpoints of the line segments. Include the first and last points
    # twice so we don't need any special syntax later to handle them.
    x: ndarray = asarray(x)
    y: ndarray = asarray(y)
    x_midpts: ndarray = hstack((x[0], 0.5 * (x[1:] + x[:-1]), x[-1]))
    y_midpts: ndarray = hstack((y[0], 0.5 * (y[1:] + y[:-1]), y[-1]))

    # Determine the start, middle, and end coordinate pair of each line segment.
    # Use the reshape to add an extra dimension so each pair of points is in its
    # own list. Then concatenate them to create:
    # [
    #   [(x1_start, y1_start), (x1_mid, y1_mid), (x1_end, y1_end)],
    #   [(x2_start, y2_start), (x2_mid, y2_mid), (x2_end, y2_end)],
    #   ...
    # ]
    coord_start: ndarray = column_stack((x_midpts[:-1], y_midpts[:-1]))[:, newaxis, :]
    coord_mid: ndarray = column_stack((x, y))[:, newaxis, :]
    coord_end: ndarray = column_stack((x_midpts[1:], y_midpts[1:]))[:, newaxis, :]
    segments: ndarray = concatenate((coord_start, coord_mid, coord_end), axis=1) # concate along new axis
    return segments
def periodic_lines(
        x: Iterable, y: Iterable, 
        domain: Iterable[Iterable], 
        alpha: Optional[float] = None
        ) -> tuple[ndarray]:
    """whenever the line x, y jumps across the domain split into a seperate sub-line"""
    assert len(x)==len(y)
    N: int = len(x)
    (xlow, xhigh), (ylow, yhigh) = domain 
    assert all(x > xlow)+all(x < xhigh)+all(y < yhigh)+all(y > ylow)
    trigger_value: float = min([abs(xhigh-xlow), abs(yhigh-ylow)]) / 2
    xs, ys = [], []
    alpha_segments = []
    j = 0
    k = 1
    while j<N:
        while not any(hypot(diff(x), diff(y))[j:k] >= trigger_value) and k < N: k+=1
        xs.append(list(x[j:k]))
        ys.append(list(y[j:k]))
        if alpha is not None: alpha_segments.append(list(alpha[j:k]))
        j+=k
        k=j+1 
    return (xs, ys) if alpha is None else (xs, ys, alpha_segments)
def plot(
        x: Iterable, y: Iterable, 
        color='black', alpha=1,
        ax: Optional[Axes] = None,
        figsize: tuple[float, float] = (1, 1),
        autoscale: bool = True,
        **kwds
        ) -> tuple[Figure, Axes, LineCollection|list[LineCollection]]:
    # prep fig, ax
    kwargs = {"capstyle":'butt', "joinstyle":'round'} | kwds
    if ax is None: fig, ax = subplots(figsize=figsize)
    fig: Figure = ax.get_figure()
    x, y = asarray(x), asarray(y)
    if type(x[0]) in Number.types:
        segs = line2segments(x, y)
        lc = LineCollection(segs, color=color, alpha=alpha, **kwargs)
        ax.add_collection(lc)
        if autoscale:
            ax.set_xlim(nanmin(x), nanmax(x))
            ax.set_ylim(nanmin(y), nanmax(y))
        return fig, ax, lc
    elif type(x[0]) in Iterable.types:
        lcc = []
        for xi, yi in zip(x, y): 
            fig,ax,lc = plot(xi, yi, color=color, alpha=alpha, ax=ax, **kwds)
            lcc.append(lc)
        if autoscale:
            ax.set_xlim(nanmin(x), nanmax(x))
            ax.set_ylim(nanmin(y), nanmax(y))
        return fig, ax, lcc 
    else: raise TypeError(f"{x[0]} not a valid number or iterable")
def _align_algorithm(
        x: Iterable, mode: str
        ) -> Iterable:
    match mode:
        case 'mid'|'m'|'center'|'c':
            return [(x[i] + x[i+1])/2 for i in range(len(x)-1)]
        case 'logmid'|'lm':
            return [log10((10**x[i] + 10**x[i+1])/ 2) for i in range(len(x)-1)]
        case 'left'|'l':
            return x[:-1]
        case 'right'|'r':
            return x[1:]
def diffplot(
        x: Iterable, y: Iterable, 
        align: str = 'mid', 
        **kwargs
        ) -> tuple[Figure, Axes, LineCollection]:
    dx: ndarray = diff(x)
    dy: ndarray = diff(y)
    x_aligned: ndarray = _align_algorithm(x, align)
    return plot(x_aligned, dy/dx, **kwargs)
def powerlawplot(
        x: Iterable, y: Iterable, 
        align: str = 'logmid', 
        **kwargs
        ) -> tuple[Figure, Axes, LineCollection]:
    return diffplot(log10(x), log10(y), align=align, **kwargs)

# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
# >-|===|>                            Decorators                           <|===|-<
# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
# >-|===|>                             Classes                             <|===|-<
# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==



