import numpy as np
from kplot.utils import auto_norm
from bokeh.plotting import figure, show
from bokeh.layouts import layout
from bokeh.models import ColumnDataSource, CustomJS, Slider, LinearColorMapper
from bokeh.io import output_notebook, curdoc

output_notebook() 
curdoc().theme = "dark_minimal"

def scrub_images(z, norm=None):
    """for scrubbing through datacubes/lists of images"""
    # init figure
    plot = figure(tools=[])
    plot.toolbar.logo = None
    plot.toolbar_location = None
    # same as matplotlib Normalize(vmin=np.nanmin(z), vmax=np.nanmax(z))
    color_mapper = LinearColorMapper(
        palette="Plasma256", #same as matplotlib plasma colormap
        low=np.nanmin(z), 
        high=np.nanmax(z)
    )
    # ColumnDataSource is bokeh's data structure
    source_visible = ColumnDataSource({
        'z':[z[0]] # whatever is in source_visible['z'] will be the image shown
    })
    source_available = ColumnDataSource({
        str(i): [z[i]] for i in range(len(z)) # container for all images
    })

    img = plot.image(
        image='z', 
        source=source_visible, # plot the image stored in source_visible['z']
        color_mapper=color_mapper, 
        x=0, y=0, # you always have to give the image an origin and width for some reason
        dw=1, dh=1
    )
    color_bar = img.construct_color_bar(padding=1)
    plot.add_layout(color_bar, "right") # adds the colorbar to the plot

    slider = Slider(
        start=0, 
        end=len(z)-1, 
        value=0, # starting value
        title='index'
    )
    callback = CustomJS( # javascript logic
        args={ # args values are available in the code using the key as the variable name
            'avail':source_available, 
            'vis':source_visible, 
            'slider':slider
        }, 
        code="""
            var new_image = avail.data[slider.value.toString()]
            vis.data['z'] = new_image 
            vis.change.emit()
        """
    )
    # whenever slider.value changes callback is called, which switches the image stored in source_visible['z']
    slider.js_on_change('value', callback)
    # layout the plot with the slider above it
    fig=layout(
        [
            [slider], 
            [plot]
        ]
    )
    # show will make it visible in a jupyter notebook, unless you call bokeh.io.output_file("file_name.html") beforehand
    show(fig)