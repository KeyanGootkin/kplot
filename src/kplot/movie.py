from kplot.image import show, column_width, two_column_width
from kplot.plot import plot 
from kplot.axes import subplots
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np

def show_movie(
        frames: np.ndarray,
        fname: str|None = None,
        fps: int = 30,
        figsize: tuple = (two_column_width, two_column_width)
):
    # create figure and axis 
    fig, ax = subplots(figsize=figsize)
    fig,ax,[img] = show(frames[0], fig=fig, ax=ax)
    
    def update(f: int):
        img.set_array(frames[f])
        ax.set_title(f"frame: {f}")

    anim = FuncAnimation(fig, update, frames=len(frames))
    anim.save(fname, fps=fps)

def line_movie(
        xs, ys,
        fname: str|None = None,
        fps: int = 30,
        figsize: tuple = (two_column_width, two_column_width)
):
    fig,ax = subplots(figsize=figsize)
    fig,ax,lines = plot(xs[0], ys[0], fig=fig, ax=ax)
    line=lines[0]

    def update(f: int):
        line.set_xdata(xs[f])
        line.set_ydata(ys[f])
        ax.set_title(f"frame: {f}")
    
    anim = FuncAnimation(fig, update, frames=len(xs))
    anim.save(fname, fps=fps)