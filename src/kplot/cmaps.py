# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
# >-|===|>                             Imports                             <|===|-<
# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
from kbasic.typing import Number, Iterable
from matplotlib.pyplot import cm
from matplotlib.colors import Colormap, LinearSegmentedColormap, ListedColormap, \
        hex2color, Normalize, LogNorm, FuncNorm, AsinhNorm, PowerNorm, SymLogNorm, \
        BoundaryNorm, CenteredNorm, TwoSlopeNorm 
from numpy import uint8, zeros, ndarray, inf, nanmin, nanmax, nanquantile, nanmean, \
        nanstd, absolute, log10, ones, linspace
from colorist import ColorOKLCH

# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
# >-|===|>                              Types                              <|===|-<
# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
class Norm:
    types: list = [
        Normalize, LogNorm, FuncNorm, AsinhNorm, PowerNorm, SymLogNorm, 
        BoundaryNorm, CenteredNorm, TwoSlopeNorm
    ]
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
colors_list = zeros((256, 4))
colors_list[:128] =  list(hex2color(manoaskies_background_blue))+[1]
for i in range(28): colors_list[128+i] = pink2grey(i/28)
for i in range(100): colors_list[156+i] = grey2black(i/150)
manoaskies_beauty = ListedColormap(colors_list)
default_cmap = cm.plasma

# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
# >-|===|>                            Functions                            <|===|-<
# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
def auto_norm(
    norm: str, 
    frames: ndarray, 
    linear_threshold: float|None = None, 
    center: float|None = None, 
    saturate: float|None = None
) -> Norm:
    """A function to create a matplotlib normalization given a set images.

    Args:
        norm (str): what type of scale to use, e.g. lognorm or centerednorm
        frames (ndarray): the images to base the normalization on
        linear_threshold (float | None, optional): for symlognorm. Defaults to None.
        center (float | None, optional): for centered normalizations. Defaults to None.
        saturate (float | None, optional): the level at which to saturate the norm, e.g. if saturate=0.01 then the 
            max is the 99th percentile. Defaults to None.

    Returns:
        matplotlib normalization
    """
    frames = frames[(-inf < frames)&(frames < inf)]
    # set min/max IF saturate is None                          or IF saturate is a tuple                                          ELSE assume its a float
    low = nanmin(frames) if saturate is None else nanquantile(frames, 1-saturate[0]) if isinstance(saturate, tuple) else nanquantile(frames, 1-saturate)
    high = nanmax(frames) if saturate is None else nanquantile(frames, 0+saturate[1]) if isinstance(saturate, tuple) else nanquantile(frames, 0+saturate)
    match norm.lower():
        case "lognorm"|"log":
            if low < 0: raise ValueError(f"minimum is {low}, LogNorm only takes positive values")
            if low==0: low=nanmin(frames[frames!=0])
            return LogNorm(vmin=low, vmax=high)
        case "symlognorm"|"symlog"|"sym":
            sig = nanstd(frames)
            mu = nanmean(frames)
            if absolute(mu)-sig > 0: raise TypeError("SymLogNorm is only designed for stuff close to zero!")
            return SymLogNorm(sig if linear_threshold is None else linear_threshold, vmin=low, vmax=high)
        case n if n in ["centerednorm", "twoslope", "twoslopenorm"]:
            sig = nanstd(frames)
            mu = nanmean(frames)
            # for the center use center if give otherwise use 0 if mean is small, else use mean
            vcenter = center if not center is None else 0 if absolute(mu)-sig > 0 else mu
            return TwoSlopeNorm(vmin=low, vcenter=vcenter, vmax=high)
        case _: return Normalize(vmin=low, vmax=high)
def align_algorithm(x: list|ndarray, mode: str):
    match mode:
        case 'mid'|'m'|'center'|'c':
            return [(x[i] + x[i+1])/2 for i in range(len(x)-1)]
        case 'logmid'|'lm':
            return [log10((10**x[i] + 10**x[i+1])/ 2) for i in range(len(x)-1)]
        case 'left'|'l':
            return x[:-1]
        case 'right'|'r':
            return x[1:]
def oklch_cmap(
    luminosity: float|Iterable[float] = (0, 1),
    chroma: float|Iterable[float] = (0, 0.4),
    hue: float|Iterable[float] = (90, 270),
    bad = None, under = None, over = None,
    N = 256
) -> ListedColormap:
    ls = ones(N) * luminosity if type(luminosity) in Number.types else linspace(*luminosity, N)
    cs = ones(N) * chroma if type(chroma) in Number.types else linspace(*chroma, N)
    hs = ones(N) * hue if type(hue) in Number.types else linspace(*hue, N) % 360
    colors_list = [
        OKLCH(li, ci, hi).rgb for li, ci, hi in zip(ls, cs, hs)
    ]
    return ListedColormap(colors_list, N=N).with_extremes(
        bad = colors_list[0] if bad is None else bad,
        under = colors_list[0] if under is None else under,
        over = colors_list[-1] if over is None else over
    )
def oklch_cmap_diverging(
    luminosity: Iterable[float] = (1, 0, 1),
    chroma: Iterable[float] = (0.4, 0, 0.4),
    hue: Iterable[float] = (90, 270),
    bad = None, under = None, over = None,
    N = 256
) -> ListedColormap:
    ls = ones(N)
    ls[:N//2] = linspace(*luminosity[:2], N//2)
    ls[N//2:] = linspace(*luminosity[1:], N//2)
    cs = ones(N)
    cs[:N//2] = linspace(*chroma[:2], N//2)
    cs[N//2:] = linspace(*chroma[1:], N//2)
    hs = linspace(*hue, N) % 360
    colors_list = [
        OKLCH(li, ci, hi).rgb for li, ci, hi in zip(ls, cs, hs)
    ]
    return ListedColormap(colors_list, N=N).with_extremes(
        bad = colors_list[N//2] if bad is None else bad,
        under = colors_list[0] if under is None else under,
        over = colors_list[-1] if over is None else over
    )

# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
# >-|===|>                             Classes                             <|===|-<
# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
class OKLCH:
    def __init__(self, lightness: float, chroma: float, hue: float, alpha: float = 1) -> None:
        assert all(m:=[0<=lightness<=1, 0.<=chroma<=0.4, 0<=hue<360, 0<=alpha<=1]), \
            f"invalid value given in {lightness, chroma, hue, alpha}"
        self.lightness = lightness 
        self.chroma = chroma 
        self.hue = hue 
        self.alpha = alpha 
    @property 
    def rgb(self) -> tuple[float]: 
        cobj = ColorOKLCH(self.lightness, self.chroma, self.hue).convert_oklch_to_srgb()
        return (uint8(cobj.red)/256, uint8(cobj.green)/256, uint8(cobj.blue)/256)
    @property 
    def rgba(self) -> tuple[float]:
        return tuple([*self.rgb, self.alpha])
    @property
    def ansi(self) -> str:
        return ColorOKLCH(self.lightness, self.chroma, self.hue).generate_ansi_code()