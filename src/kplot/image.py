# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
# >-|===|>                             Imports                             <|===|-<
# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
from kplot.cmaps import Cmap
from kplot.axes import subplots
from typing import Optional
from numpy import ndarray, mgrid, asarray
from matplotlib.figure import Figure 
from matplotlib.axes import Axes
from matplotlib.pyplot import cm, savefig
from matplotlib.pyplot import show as show_fig
from mpl_toolkits.axes_grid1 import make_axes_locatable

# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
# >-|===|>                            Functions                            <|===|-<
# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
def show(
        image: ndarray,
        #x/y axes
        x: Optional[ndarray] = None,
        y: Optional[ndarray] = None,
        #figure setup
        fig: Optional[Figure] = None,
        ax: Optional[Axes] = None,
        axis: bool = True,
        figsize: tuple[float, float] = (10, 10),
        show: bool = False,
        #plot parameters
        cmap: str|Cmap = cm.plasma,
        colorbar: bool = True,
        colorbar_style: dict = {'location':'right', "size":"7%", "pad":0.05},
        cticks: Optional[list] = None,
        units: Optional[str] = None,
        #contour options
        contour: Optional[ndarray] = None,
        contour_style: dict = {'levels': 10, 'colors': 'black'},
        #plot formating
        xlim: tuple = (None, None),
        ylim: tuple = (None, None),
        title: Optional[str] = None,
        #presentation parameters
        save: str = "",
        dpi: int = 100,
        #everything else goes into pcolormesh
        **kwargs
):
    """a convinient way to imshow and arrange it nicely

    Args:
        field (ndarray): The image to be plotted
        x (ndarray, optional): x coordinates of pixels. Defaults to None.
        y (ndarray, optional): y coordinates of pixels. Defaults to None.
        fig (Figure, optional): the figure on which to plot. Defaults to None.
        ax (Axes, optional): the ax on which to plot this image. Defaults to None.
        axis (bool, optional): whether to include the whole axis, if False then only the plot area will be shown.
        figsize (tuple[float, float], optional): unless fig and ax are given plot on a figure of this size. Defaults to (10, 10).
        show (bool, optional): whether to use the show() command at the end. Defaults to True.
        cmap (Cmap, optional): the colormap to use for the image. Defaults to plasma.
        colorbar (bool, optional): whether to add a colorbar. Defaults to True.
        colorbar_style (dict, optional): dictionary containing kwargs for the colorbar command. Defaults to {'location':'right', "size":"7%", "pad":0.05}.
        cticks (list, optional): the ticks to use on the colorbar. Defaults to None.
        units (str, optional): label for the colorbar. Defaults to None.
        contour (ndarray, optional): whether to put a contour plot on top. Defaults to None.
        contour_style (dict, optional): kwargs for contour plotting command. Defaults to {'levels': 10, 'colors': 'black'}.
        xlim (tuple, optional): the limits on the x axis. Defaults to (None, None).
        ylim (tuple, optional): the limits on the y axis. Defaults to (None, None).
        title (str, optional): the plot title. Defaults to None.
        save (str, optional): if given save the plot to this path. Defaults to "".
        dpi (int, optional): dots per inch to save at. Defaults to 100.

    Returns:
        fig (Figure): The figure on which the plot was put
        ax (Axes): The ax on which the plot was put
        img (Artist): the plot artist
    """
    # prep data
    assert (ndims:=len(image.shape))==2, f"show was given an image with {ndims} dimensions, please provide a 2d array"
    # check axes
    if x is None: 
        x, y = mgrid[:image.shape[0], :image.shape[1]]
    else: 
        x, y = asarray(x), asarray(y)
        assert (len(x), len(y)) == image.shape, f"Given x of shape {len(x)} and y of shape {len(y)} but image is of shape {image.shape}"
    # prep figure
    if ax is None: (fig, ax) = subplots(figsize=figsize)
    fig = ax.get_figure()
    # see if we want a frame
    if not axis: 
        ax.axis('off')
        ax.set_position([0, 0, 1, 1])
        colorbar = False
    # plot data
    img = ax.pcolormesh(x, y, image, cmap=cmap, **kwargs)
    # colorbar
    if colorbar: 
        divider = make_axes_locatable(ax)
        colorbar_location = colorbar_style.pop("location") if "location" in colorbar_style.keys() else "right"
        cax = divider.append_axes(colorbar_location, **colorbar_style)
        fig.colorbar(img, cax=cax, ax=ax, ticks=cticks, label=units, orientation = 'vertical' if colorbar_location in ('right', 'left') else 'horizontal')
        if colorbar_location=='top':
            cax.xaxis.set_ticks_position('top')
            cax.xaxis.set_label_position('top')
    # contour
    if not contour is None: ax.contour(contour, **contour_style)
    # set limits
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    # set title
    ax.set_title(title)
    # set aspect
    ax.set_aspect('equal')
    # present the figure
    if len(save)>0: 
        if not '.' in save: save +=".jpeg"
        savefig(save, dpi=dpi, bbox_inches='tight')
    if show: show_fig()
    return fig, ax, img
def contour(
        Z: ndarray, 
        #x/y axes
        x: ndarray|None = None,
        y: ndarray|None = None,
        #figure setup
        fig = None,
        ax = None,
        axis: bool = True,
        figsize: tuple[float, float] = (10, 10),
        show: bool = False,
        #plot parameters
        levels = 10,
        color = 'black',
        colors = None,
        cmap = None,
        negative_linestyles = '-',
        linestyles = '-',
        #plot formating
        xlim: tuple = (None, None),
        ylim: tuple = (None, None),
        title: str|None = None,
        #presentation parameters
        save: str = "",
        dpi: int = 100,
        #throw the rest in a dict
        **kwds
    ):
    # prep data
    assert (ndims:=len(Z.shape))==2, f"show was given an image with {ndims} dimensions, please provide a 2d array"
    # check axes
    if x is None: x, y = mgrid[:Z.shape[0], :Z.shape[1]]
    else: assert (len(x), len(y)) == Z.shape, f"Given x of shape {len(x)} and y of shape {len(y)} but image is of shape {Z.shape}"
    # prep figure
    if ax is None: (fig, ax) = subplots(figsize=figsize)
    fig = ax.get_figure()
    if not axis: 
        ax.axis('off')
        ax.set_position([0, 0, 1, 1])
        colorbar = False
    # plot data
    contour_dict = {'levels':levels,"linestyles":linestyles,'negative_linestyles':negative_linestyles}
    if cmap: #plot with a cmap
        contour_dict['cmap'] = cmap
    else: #plot with a color
        contour_dict['colors'] = color if not colors else colors
    img = ax.contour(
        x, y, Z,
        **contour_dict,
        **kwds
    )
    # set limits
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    # set title
    ax.set_title(title)
    # set aspect
    ax.set_aspect('equal')
    # present the figure
    if len(save)>0: 
        if not '.' in save: save +=".jpeg"
        savefig(save, dpi=dpi)
    if show: show_fig()
    return fig, ax, img
def contourf(
        Z: ndarray, 
        #x/y axes
        x: ndarray|None = None,
        y: ndarray|None = None,
        #figure setup
        fig = None,
        ax = None,
        axis: bool = True,
        figsize: tuple[float, float] = (10, 10),
        show: bool = False,
        #plot parameters
        levels = 10,
        cmap = cm.plasma,
        colorbar: bool = True,
        colorbar_style: dict = {'location':'right', "size":"7%", "pad":0.05},
        cticks: list|None = None,
        units: str|None = None,
        #plot formating
        xlim: tuple = (None, None),
        ylim: tuple = (None, None),
        title: str|None = None,
        #presentation parameters
        save: str = "",
        dpi: int = 100,
        #throw the rest in a dict
        **kwds
    ):
    # prep data
    assert (ndims:=len(Z.shape))==2, f"show was given an image with {ndims} dimensions, please provide a 2d array"
    # check axes
    if x is None: x, y = mgrid[:Z.shape[0], :Z.shape[1]]
    else: assert (len(x), len(y)) == Z.shape, f"Given x of shape {len(x)} and y of shape {len(y)} but image is of shape {Z.shape}"
    # prep figure
    if ax is None: (fig, ax) = subplots(figsize=figsize)
    fig = ax.get_figure()
    if not axis: 
        ax.axis('off')
        ax.set_position([0, 0, 1, 1])
        colorbar = False
    # plot data
    contour_dict = {'levels':levels,"cmap":cmap}
    img = ax.contourf(
        x, y, Z,
        **contour_dict,
        **kwds
    )
    # colorbar
    if colorbar: 
        divider = make_axes_locatable(ax)
        colorbar_location = colorbar_style.pop("location") if "location" in colorbar_style.keys() else "right"
        cax = divider.append_axes(colorbar_location, **colorbar_style)
        fig.colorbar(img, cax=cax, ax=ax, ticks=cticks, label=units)
    # set limits
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    # set title
    ax.set_title(title)
    # set aspect
    ax.set_aspect('equal')
    # present the figure
    if len(save)>0: 
        if not '.' in save: save +=".jpeg"
        savefig(save, dpi=dpi)
    if show: show_fig()
    return fig, ax, img
