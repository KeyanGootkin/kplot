# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
# >-|===|>                             Imports                             <|===|-<
# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
from kplot.utils import column_width, two_column_width
from kplot.cmaps import auto_norm
from kplot.image import show
from kbasic.bar import verbose_bar
from kbasic.parsing import ensure_path
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
from os import system
from glob import glob
from typing import Callable

# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
# >-|===|>                              Types                              <|===|-<
# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
# >-|===|>                           Definitions                           <|===|-<
# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
# >-|===|>                            Functions                            <|===|-<
# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
def ffmpeg(
        file_name: str, 
        source: str = "./frames", destination: str = "./", 
        fps: int = 30
    ) -> None:
    file_path: str = destination + '/' + file_name
    if not file_path.endswith('.mp4'): file_path += ".mp4"
    system(f"ffmpeg -loglevel 8 -framerate {fps} -pattern_type glob -i '{source+'/*.png'}' -c:v libx264 -pix_fmt yuv420p -y {file_path}")
def func_video(
        video_name: str, fig: Figure, updater: Callable, N: int, 
        frames: str = "./frames", destination: str = "./", 
        dpi: int = 100, fps: int = 30, verbose=True
    ) -> None:
    ensure_path(frames)
    if len(glob(f"{frames}/*.png"))>0: system(f"rm {frames}/*.png")
    ndigits = len(str(N))
    # if the size isn't divisible by 2 ffmpeg gets mad???
    [wpix, hpix] = (np.array(fig.get_size_inches()) * dpi // 1).astype(int)
    if wpix%2==1: wpix += 1 
    if hpix%2==1: hpix += 1
    fig.set_size_inches(wpix/dpi, hpix/dpi)
    # make the frames and save them to the frames directory
    for i in verbose_bar(range(N), verbose):
        updater(i)
        fig.savefig(f"{frames}/{str(i).zfill(ndigits)}.png", dpi=dpi)
    # make the video
    ffmpeg(video_name, source=frames, destination=destination, fps=fps)
def line_video(
    xs, ys, 
    fname, 
    destination: str = '.',
    ax=None, figsize=(5, 5),
    fps=20, dpi=100,
    **kwargs
):
    if not ax: fig, ax = plt.subplots(figsize=figsize)
    fig = ax.get_figure()
    [line] = ax.plot(xs[0], ys[0], **kwargs)
    def update(f: int):
        line.set_data(xs[f], ys[f])
    func_video(fname, fig, update, len(xs), fps=fps, dpi=dpi, destination=destination)
def lines_video(
        data,
        fname,
        destination: str = '.', frames='./frames',
        ax=None, figsize=(5,5),
        fps=20, dpi=100
) -> None:
    if not ax: fig, ax = plt.subplots(figsize=figsize)
    fig = ax.get_figure()
    lines = []
    for (x, y) in data:
        line, = ax.plot(x[0], y[0])
        lines.append(line)
    def update(i: int) -> None:
        for (x, y), line in zip(data, lines):
            line.set_data(x[i], y[i])
    func_video(fname, fig, update, min([len(data[i][0]) for i in range(len(data))]), fps=fps, dpi=dpi, destination=destination, frames=frames)
def show_video(
    frames: np.ndarray,
    fname: str,
    ax = None,
    norm = 'linear',
    destination: str = '.',
    fps: int = 30,
    dpi=100,
    **kwargs
):
    # create figure and axis 
    if isinstance(norm, str): norm = auto_norm(norm, frames)
    fig,ax,img = show(frames[0], ax=ax, norm=norm, show=False, **kwargs)
    def update(f: int):
        img.set_array(frames[f])
    func_video(fname, fig, update, len(frames), destination=destination, fps=fps, dpi=dpi)
    
# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
# >-|===|>                            Decorators                           <|===|-<
# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
def line_video_function(func):
    @wraps(func)
    def line_video_wrapper(*args, ax=None, save="default.gif", **kwargs):
        # Calculate data via func
        xs, ys = func(*args, **kwargs)
        line_video(xs, ys, save, ax=ax)
    return line_video_wrapper
def show_video_function(func):
    @wraps(func)
    def simple_video_wrapper(*args, save="default.gif", norm='linear', **kwargs):
        frames = func(*args, **kwargs)
        show_video(frames, save, norm=norm)
    return simple_video_wrapper

# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
# >-|===|>                             Classes                             <|===|-<
# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==