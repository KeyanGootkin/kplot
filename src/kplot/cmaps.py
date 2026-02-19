# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
# >-|===|>                             Imports                             <|===|-<
# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
from kbasic.array import tile
from functools import wraps
from glob import glob
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib.colors import Colormap, LinearSegmentedColormap, ListedColormap, hex2color, Normalize, LogNorm, FuncNorm, AsinhNorm, PowerNorm, SymLogNorm, BoundaryNorm, CenteredNorm, TwoSlopeNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
# >-|===|>                              Types                              <|===|-<
# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
class Norm:
    types: list = [Normalize, LogNorm, FuncNorm, AsinhNorm, PowerNorm, SymLogNorm, BoundaryNorm, CenteredNorm, TwoSlopeNorm]
class Cmap:
    types: list = [Colormap, ListedColormap, LinearSegmentedColormap]
# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
# >-|===|>                           Definitions                           <|===|-<
# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
pink = "#E34F68"
lightpink = "#E39FAA"
blue = "#7350E6"
lightblue = "#AE9FE3"
shadow = "#B8B7B8"
manoaskies = LinearSegmentedColormap.from_list("manoaskies", [pink, blue])
manoaskies_centered = LinearSegmentedColormap.from_list("manoaskies_centered", [lightpink, pink, "#000000", blue, lightblue])
manoaskies_background_blue = "#0C0524"
pink2grey = LinearSegmentedColormap.from_list("p2g", [pink, shadow])
grey2black = LinearSegmentedColormap.from_list("g2b", [shadow, "#000000"])
colors_list = np.zeros((256, 4))
colors_list[:128] =  list(hex2color(manoaskies_background_blue))+[1]
for i in range(28): colors_list[128+i] = pink2grey(i/28)
for i in range(100): colors_list[156+i] = grey2black(i/150)
manoaskies_beauty = ListedColormap(colors_list)
default_cmap = cm.plasma

# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
# >-|===|>                            Functions                            <|===|-<
# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
# # Ploting utils
def auto_norm(
    norm: str, 
    frames: np.ndarray, 
    linear_threshold: float|None = None, 
    center: float|None = None, 
    saturate: float|None = None
) -> Norm:
    """A function to create a matplotlib normalization given a set images.

    Args:
        norm (str): what type of scale to use, e.g. lognorm or centerednorm
        frames (np.ndarray): the images to base the normalization on
        linear_threshold (float | None, optional): for symlognorm. Defaults to None.
        center (float | None, optional): for centered normalizations. Defaults to None.
        saturate (float | None, optional): the level at which to saturate the norm, e.g. if saturate=0.01 then the 
            max is the 99th percentile. Defaults to None.

    Returns:
        matplotlib normalization
    """
    frames = frames[(-np.inf < frames)&(frames < np.inf)]
    # set min/max IF saturate is None                          or IF saturate is a tuple                                          ELSE assume its a float
    low = np.nanmin(frames) if saturate is None else np.nanquantile(frames, 1-saturate[0]) if isinstance(saturate, tuple) else np.nanquantile(frames, 1-saturate)
    high = np.nanmax(frames) if saturate is None else np.nanquantile(frames, 0+saturate[1]) if isinstance(saturate, tuple) else np.nanquantile(frames, 0+saturate)
    match norm.lower():
        case "lognorm"|"log":
            if low < 0: raise ValueError(f"minimum is {low}, LogNorm only takes positive values")
            if low==0: low=np.nanmin(frames[frames!=0])
            return LogNorm(vmin=low, vmax=high)
        case "symlognorm"|"symlog"|"sym":
            sig = np.nanstd(frames)
            mu = np.nanmean(frames)
            if np.abs(mu)-sig > 0: raise TypeError("SymLogNorm is only designed for stuff close to zero!")
            return SymLogNorm(sig if linear_threshold is None else linear_threshold, vmin=low, vmax=high)
        case n if n in ["centerednorm", "twoslope", "twoslopenorm"]:
            sig = np.nanstd(frames)
            mu = np.nanmean(frames)
            # for the center use center if give otherwise use 0 if mean is small, else use mean
            vcenter = center if not center is None else 0 if np.abs(mu)-sig > 0 else mu
            return TwoSlopeNorm(vmin=low, vcenter=vcenter, vmax=high)
        case _: return Normalize(vmin=low, vmax=high)
def align_algorithm(x: list|np.ndarray, mode: str):
    match mode:
        case 'mid'|'m'|'center'|'c':
            return [(x[i] + x[i+1])/2 for i in range(len(x)-1)]
        case 'logmid'|'lm':
            return [np.log10((10**x[i] + 10**x[i+1])/ 2) for i in range(len(x)-1)]
        case 'left'|'l':
            return x[:-1]
        case 'right'|'r':
            return x[1:]

# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
# >-|===|>                            Decorators                           <|===|-<
# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
