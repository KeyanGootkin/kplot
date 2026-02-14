import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes 
import numpy as np

from kplot.utils import column_width, two_column_width

def decode_subplots_args(*args, **kwargs):
    if 'figsize' in kwargs: kwargs['figsize'] = tuple(
        column_width if kwargs['figsize'][i]==1 else two_column_width if kwargs['figsize'][i]==2 else kwargs['figsize'][i]
    for i in range(2))
    match args:
        case (int(), int())|((int(), int())):
            return plt.subplots(*args, **kwargs)
        case (str(),):
            return plt.subplot_mosaic(*args, **kwargs)
        case ():
            return plt.subplots(**kwargs)

def subplots(*args, **kwargs): return decode_subplots_args(*args, **kwargs)

def access_subplots(
        fig: None|Figure = None, 
        axes: None|Axes|list = None,
        figsize: None|tuple = None
    ) -> tuple[Figure, list]:
    match fig, axes:
        # if given nothing
        case None, None:
            fig, axes = subplots(figsize=figsize)
            axes = [axes]
        case Figure(), None:
            axes = fig.get_axes()
            axes = axes if len(axes) > 0 else [fig.add_subplot(111)]
        case None, Axes():
            fig = axes.get_figure()
            axes = [axes]
        case None, list()|np.ndarray():
            axes = axes.flatten()
            fig = axes[0].get_figure()
        case Figure(), Axes():
            axes = [axes]
        case Figure(), list()|np.ndarray():
            return fig, axes  
    return fig, axes