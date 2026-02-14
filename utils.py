import numpy as np
from matplotlib.colors import LogNorm, SymLogNorm, TwoSlopeNorm, Normalize

column_width = 242.26653/72.27
two_column_width = 513.1174/72.27

def alias_kwarg(kw1, it1, kw2, it2): 
    assert not all([it1 is not None, it2 is not None]), f"either give an {kw1} or a {kw2} keyword, not both"
    return it1 if it1 is not None else it2

def parse_multiax_params(param, target_types: list, N: int, out_ndim: int = 0):
    if type(param) in [list, np.ndarray]:
        if target_types[0] in [list, np.ndarray]:
            if np.array(param).ndim == out_ndim: return [param] * N
            elif len(param) == N: return param 
            elif len(param) == 1: return [param[0]] * N 
        elif type(param[0]) in target_types:
            if len(param) == N: return param 
            elif len(param) == 1: return [param[0]] * N 
        elif type(param[0]) in [list, np.ndarray]:
            if len(param) == 1 and len(param[0]) == N: return param[0]
            elif len(param) == 1 and len(param[0]) == 1: return [param[0][0]] * N
    
    elif type(param) in target_types: return [param] * N
    return [None] * N

def auto_norm(
    norm: str, 
    frames: np.ndarray, 
    linear_threshold: float|None = None, 
    center: float|None = None, 
    saturate: float|None = None
):
    frames = frames[(-np.inf < frames)&(frames < np.inf)]
    # set min/max IF saturate is None                          or IF saturate is a tuple                                          ELSE assume its a float
    low = np.nanmin(frames) if saturate is None else np.nanquantile(frames, 1-saturate[0]) if isinstance(saturate, tuple) else np.nanquantile(frames, 1-saturate)
    high = np.nanmax(frames) if saturate is None else np.nanquantile(frames, 0+saturate[1]) if isinstance(saturate, tuple) else np.nanquantile(frames, 0+saturate)
    match norm.lower():
        case "lognorm":
            if low < 0: raise ValueError(f"minimum is {low}, LogNorm only takes positive values")
            if low==0: low=np.nanmin(frames[frames!=0])
            return LogNorm(vmin=low, vmax=high)
        case "symlognorm":
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

    
