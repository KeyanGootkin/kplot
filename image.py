import matplotlib.pyplot as plt
import numpy as np
from kplot.Axes import access_subplots
from kplot.utils import alias_kwarg, parse_multiax_params, column_width, two_column_width
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, Colormap
CmapTypes = [ListedColormap, LinearSegmentedColormap, Colormap]
from matplotlib.colors import Normalize, LogNorm, FuncNorm, AsinhNorm, PowerNorm, SymLogNorm, BoundaryNorm, CenteredNorm, TwoSlopeNorm
NormTypes = [Normalize, LogNorm, FuncNorm, AsinhNorm, PowerNorm, SymLogNorm, BoundaryNorm, CenteredNorm, TwoSlopeNorm]



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
    cmaps = parse_multiax_params(plot_dict['cmap'], CmapTypes, N)
    colorbars = 'single' if cbar_dict['colorbar']=='single' else parse_multiax_params(cbar_dict['colorbar'], [bool], N)
    norms = 'single' if colorbars=='single' else parse_multiax_params(plot_dict['norm'], NormTypes, N)
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
        src,
        #x/y axes
        x: np.ndarray|None = None,
        y: np.ndarray|None = None,
        #figure setup
        fig=None, axes=None, ax=None,
        figsize: tuple[float, float] = (column_width, column_width),
        show: bool = True,
        #plot color parameters
        cmap = plt.cm.plasma,
        norm = None,
        colorbar: bool = True,
        cbar_location: str = 'right',
        cbar_size: str = "5%",
        cbar_pad: float = 0.05,
        cticks: list|None = None,
        units: str|None = None,
        #plot formating
        xlabel: None|str = None,
        ylabel: None|str = None, 
        xlim: tuple = (None, None),
        ylim: tuple = (None, None),
        title: str|None = None,
        aspect: str = 'equal',
        #presentation parameters
        save: str = "",
        dpi: int = 100,
        #everything else goes into pcolormesh
        **kwargs
    ):
    # set up the image
    image = decode_image_src(src)
    # prep figure
    close_fig = True if fig is None else False
    show = show if fig is None else False
    axes = alias_kwarg("axes", axes, "ax", ax)
    fig, axes = access_subplots(fig=fig, axes=axes, figsize=figsize)
    # check axes match image
    if x is None: x, y = np.mgrid[:image.shape[-2], :image.shape[-1]]
    # prepare to pass info to plotting functions
    plot_dict = {
        'x': x,
        'y': y,
        'image': image,
        'fig': fig,
        'axes': axes,
        'cmap': cmap,
        'norm': norm
    }
    cbar_dict = {
        'colorbar': colorbar, 
        'location': cbar_location,
        'size': cbar_size,
        'pad': cbar_pad,
        'cticks': cticks, 
        'units': units
    }
    axes_dict = {
        'xlim': xlim,
        'ylim': ylim,
        'xlabel': xlabel,
        'ylabel': ylabel,
        'title': title,
        'aspect': aspect
    }
    match image.ndim:
        case 2: 
            imgs = _show_one_frame(plot_dict, axes_dict, cbar_dict, **kwargs)
        case 3:
            multiax = image.shape[0] == len(axes)
            assert multiax, f"Can only handle multiple images if there are multiple axes, {image.shape, len(axes)}"
            imgs = _show_multiple_frames(plot_dict, axes_dict, cbar_dict, N=len(image), **kwargs)

    # present the figure
    if len(save)>0: 
        if not '.' in save: save +=".jpg"
        plt.savefig(save, dpi=dpi)
    if show: plt.show()
    if close_fig: plt.close(fig)
    else: return fig, ax, imgs
