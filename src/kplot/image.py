# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
# >-|===|>                             Imports                             <|===|-<
# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
from kplot.axes import access_subplots
from kplot.utils import alias_kwarg, parse_multiax_params, column_width, two_column_width
from kplot.cmaps import Norm, Cmap

from kbasic.array import tile
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, Colormap

# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
# >-|===|>                              Types                              <|===|-<
# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
# >-|===|>                           Definitions                           <|===|-<
# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
# >-|===|>                            Functions                            <|===|-<
# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
def decode_image_src(src) -> np.ndarray:
    match src:
        case str(): 
            return np.loadtxt(src)
        case np.ndarray():
            return src
        case list():
            return np.array(src)
def _show_one_frame(plot_dict, axes_dict, cbar_dict, **kwargs):
    # populate name space
    x, y, image = plot_dict['x'], plot_dict['y'], plot_dict['image']
    fig, ax, cmap, norm = plot_dict['fig'], plot_dict['axes'][0], plot_dict['cmap'], plot_dict['norm']
    # plot
    img = ax.pcolormesh(x, y, image, cmap=cmap, norm=norm, **kwargs)
    # label axes
    ax.set_xlabel(axes_dict['xlabel'])
    ax.set_ylabel(axes_dict['ylabel'])
    # set limits
    ax.set_xlim(axes_dict['xlim'])
    ax.set_ylim(axes_dict['ylim'])
    # set title
    ax.set_title(axes_dict['title'])
    # set aspect
    ax.set_aspect(axes_dict['aspect'])
    if cbar_dict['colorbar']: 
        divider = make_axes_locatable(ax)
        cax = divider.append_axes(
            cbar_dict['location'], 
            size=cbar_dict['size'], 
            pad=cbar_dict['pad']
        )
        fig.colorbar(
            img, 
            cax=cax, ax=ax, 
            ticks=cbar_dict['cticks'], 
            label=cbar_dict['units']
        )
    return [img]
def construct_mass_norm(images: np.ndarray, norm):
    match norm:
        # linear norms
        case Normalize():
            vmin = np.nanmin(images) if norm.vmin is None else norm.vmin 
            vmax = np.nanmax(images) if norm.vmax is None else norm.vmax 
            return Normalize(vmin=vmin, vmax=vmax, clip=norm.clip)
            
        case CenteredNorm():
            vmin = np.nanmin(images) if norm.vmin is None else norm.vmin 
            vmax = np.nanmax(images) if norm.vmax is None else norm.vmax
            # only center at 0 if the mean is close to 0 to avoid weird balance
            vcen = norm.vcenter if norm.vcenter else 0 if np.nanmean(images) - np.nanstd(images)/2 else np.nanmean(images)
            hrng = norm.halfrange if norm.halfrange else vmax/2 if vcen==0 else abs(vmax-vmin)/2
            return CenteredNorm(vcenter=vcen, halfrange=hrng, clip=norm.clip)
        
        case TwoSlopeNorm():
            vmin = np.nanmin(images) if norm.vmin is None else norm.vmin 
            vmax = np.nanmax(images) if norm.vmax is None else norm.vmax
            vcen = norm.vcenter if norm.vcenter else 0 if np.nanmean(images) - np.nanstd(images)/2 else np.nanmean(images)
            return TwoSlopeNorm(vcenter=vcen, vmin=vmin, vmax=vmax)
        
        case BoundaryNorm():
            return BoundaryNorm(norm.boundaries, norm.ncolors, clip=norm.clip, extend=norm.extend)
        
        # log-like norms
        case LogNorm():
            vmin = np.nanmin(images) if norm.vmin is None else norm.vmin 
            vmax = np.nanmax(images) if norm.vmax is None else norm.vmax
            return LogNorm(vmin=vmin, vmax=vmax, clip=norm.clip)
            
        case SymLogNorm():
            vmin = np.nanmin(images) if norm.vmin is None else norm.vmin 
            vmax = np.nanmax(images) if norm.vmax is None else norm.vmax
            #check that this isn't trivially the same as LogNorm
            if vmin > 0: return LogNorm(vmin=vmin, vmax=vmax, clip=norm.clip)
            linthresh = np.nanquantile(images, .1) if norm.linthresh is None else norm.linthresh
            return SymLogNorm(linthresh, linscale=norm.linscale, vmin=vmin, vmax=vmax, clip=norm.clip)
        
        case AsinhNorm():
            vmin = np.nanmin(images) if norm.vmin is None else norm.vmin 
            vmax = np.nanmax(images) if norm.vmax is None else norm.vmax
            linwidth = np.nanquantile(images, .1) if norm.linear_width is None else norm.linear_width
            return AsinhNorm(linear_width=linwidth, vmin=vmin, vmax=vmax, clip=norm.clip)
                    
        case PowerNorm():
            vmin = np.nanmin(images) if norm.vmin is None else norm.vmin 
            vmax = np.nanmax(images) if norm.vmax is None else norm.vmax
            return PowerNorm(norm.gamma, vmin=vmin, vmax=vmax, clip=norm.clip)
        
        case FuncNorm():
            vmin = np.nanmin(images) if norm.vmin is None else norm.vmin 
            vmax = np.nanmax(images) if norm.vmax is None else norm.vmax
            return FuncNorm(norm.functions, vmin=vmin, vmax=vmax, clip=norm.clip)

    pass
def _show_multiple_frames(plot_dict: dict, axes_dict: dict, cbar_dict: dict, N: int, **kwargs):
    fig, axes, images = plot_dict['fig'], plot_dict['axes'], plot_dict['image'] # these are set as single and multiple before arriving to this function    
    xs = parse_multiax_params(plot_dict['x'], [np.ndarray], N)
    ys = parse_multiax_params(plot_dict['y'], [np.ndarray], N)
    cmaps = parse_multiax_params(plot_dict['cmap'], Cmap.types, N)
    colorbars = 'single' if cbar_dict['colorbar']=='single' else parse_multiax_params(cbar_dict['colorbar'], [bool], N)
    norms = 'single' if colorbars=='single' else parse_multiax_params(plot_dict['norm'], Norm.types, N)
    cbar_locations = parse_multiax_params(cbar_dict['location'], [str], N)
    cbar_sizes = parse_multiax_params(cbar_dict['size'], [str], N)
    cbar_pads = parse_multiax_params(cbar_dict['pad'], [float], N)
    cticks = parse_multiax_params(cbar_dict['cticks'], [list], N)
    units = parse_multiax_params(cbar_dict['units'], [str], N)
    xlims = parse_multiax_params(axes_dict['xlim'], [tuple], N)
    ylims = parse_multiax_params(axes_dict['ylim'], [tuple], N)
    xlabels = 'single' if type(axes_dict['xlabel'])==str else parse_multiax_params(axes_dict['xlabel'], [str], N)
    ylabels = 'single' if type(axes_dict['ylabel'])==str else parse_multiax_params(axes_dict['ylabel'], [str], N)
    titles = 'single' if type(axes_dict['title'])==str else parse_multiax_params(axes_dict['title'], [str], N)
    aspects = parse_multiax_params(axes_dict['aspect'], [str], N)
    imgs = []
    for i in range(N):
        ax, image = axes[i], images[i]
        imgs.append(ax.pcolormesh(xs[i], ys[i], image, cmap=cmaps[i], norm=norms[i], **kwargs))
    return imgs
def show(
        field: np.ndarray,
        tile_image: bool = False,
        #x/y axes
        x: np.ndarray|None = None,
        y: np.ndarray|None = None,
        #figure setup
        fig = None,
        ax = None,
        axis: bool = True,
        figsize: tuple[float, float] = (10, 10),
        show: bool = False,
        #plot parameters
        cmap = plt.cm.plasma,
        colorbar: bool = True,
        colorbar_style: dict = {'location':'right', "size":"7%", "pad":0.05},
        cticks: list|None = None,
        units: str|None = None,
        #contour options
        contour: np.ndarray|None = None,
        contour_style: dict = {'levels': 10, 'colors': 'black'},
        #plot formating
        xlim: tuple = (None, None),
        ylim: tuple = (None, None),
        title: str|None = None,
        #presentation parameters
        save: str = "",
        dpi: int = 100,
        #everything else goes into pcolormesh
        **kwargs
):
    """a convinient way to plt.imshow and arrange it nicely

    Args:
        field (np.ndarray): The image to be plotted
        tile_image (bool, optional): whether to tile the image. Defaults to False.
        x (np.ndarray | None, optional): x coordinates of pixels. Defaults to None.
        y (np.ndarray | None, optional): y coordinates of pixels. Defaults to None.
        fig (plt.Figure, optional): the figure on which to plot. Defaults to None.
        ax (plt.Axes, optional): the ax on which to plot this image. Defaults to None.
        axis (bool, optional): whether to include the whole axis, if False then only the plot area will be shown.
        figsize (tuple[float, float], optional): unless fig and ax are given plot on a figure of this size. Defaults to (10, 10).
        show (bool, optional): whether to use the plt.show() command at the end. Defaults to True.
        cmap (plt.ColorMap, optional): the colormap to use for the image. Defaults to plasma.
        colorbar (bool, optional): whether to add a colorbar. Defaults to True.
        colorbar_style (dict, optional): dictionary containing kwargs for the colorbar command. Defaults to {'location':'right', "size":"7%", "pad":0.05}.
        cticks (list | None, optional): the ticks to use on the colorbar. Defaults to None.
        units (str | None, optional): label for the colorbar. Defaults to None.
        contour (np.ndarray | None, optional): whether to put a contour plot on top. Defaults to None.
        contour_style (dict, optional): kwargs for contour plotting command. Defaults to {'levels': 10, 'colors': 'black'}.
        xlim (tuple, optional): the limits on the x axis. Defaults to (None, None).
        ylim (tuple, optional): the limits on the y axis. Defaults to (None, None).
        title (str | None, optional): the plot title. Defaults to None.
        save (str, optional): if given save the plot to this path. Defaults to "".
        dpi (int, optional): dots per inch to save at. Defaults to 100.

    Returns:
        fig (plt.Figure): The figure on which the plot was put
        ax (plt.Axes): The ax on which the plot was put
        img (plt.Artist): the plot artist
    """
    # prep data
    assert (ndims:=len(field.shape))==2, f"show was given an image with {ndims} dimensions, please provide a 2d array"
    image = tile(field) if tile_image else field
    # check axes
    if x is None: x, y = np.mgrid[:image.shape[0], :image.shape[1]]
    if tile_image: x, y = np.r_[x, x+x[-1], x+2*x[-1]], np.r_[y, y+y[-1], y+2*y[-1]]
    else: assert (len(x), len(y)) == image.shape, f"Given x of shape {len(x)} and y of shape {len(y)} but image is of shape {image.shape}"
    # prep figure
    if ax is None: (fig, ax) = plt.subplots(figsize=figsize)
    fig = ax.get_figure()
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
        plt.savefig(save, dpi=dpi, bbox_inches='tight')
    if show: plt.show()
    return fig, ax, img
def contour(
        Z: np.ndarray, 
        #x/y axes
        x: np.ndarray|None = None,
        y: np.ndarray|None = None,
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
    if x is None: x, y = np.mgrid[:Z.shape[0], :Z.shape[1]]
    else: assert (len(x), len(y)) == Z.shape, f"Given x of shape {len(x)} and y of shape {len(y)} but image is of shape {Z.shape}"
    # prep figure
    if ax is None: (fig, ax) = plt.subplots(figsize=figsize)
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
        plt.savefig(save, dpi=dpi)
    if show: plt.show()
    return fig, ax, img
def contourf(
        Z: np.ndarray, 
        #x/y axes
        x: np.ndarray|None = None,
        y: np.ndarray|None = None,
        #figure setup
        fig = None,
        ax = None,
        axis: bool = True,
        figsize: tuple[float, float] = (10, 10),
        show: bool = False,
        #plot parameters
        levels = 10,
        cmap = plt.cm.plasma,
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
    if x is None: x, y = np.mgrid[:Z.shape[0], :Z.shape[1]]
    else: assert (len(x), len(y)) == Z.shape, f"Given x of shape {len(x)} and y of shape {len(y)} but image is of shape {Z.shape}"
    # prep figure
    if ax is None: (fig, ax) = plt.subplots(figsize=figsize)
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
        plt.savefig(save, dpi=dpi)
    if show: plt.show()
    return fig, ax, img

# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
# >-|===|>                            Decorators                           <|===|-<
# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
# >-|===|>                             Classes                             <|===|-<
# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==