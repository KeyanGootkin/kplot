# kplot
Hey all, this is just a tiny little package to change how matplotlib looks :-)

## Installation
I recommend using pip
```Unix
pip install kplot
```
or to update it
```Unix
pip install kplot -U
```
If you want to get fancy just make sure you put kplot somewhere in your path, since kplot.dark, and kplot.kg look there for the style sheets!

## Usage
To change how your plots look all you have to do is add a simple import statement!
```Python
import kplot.kg                #a pretty default light style
#or
import kplot.dark              #a dark style, perfect for Jupyter Lab's dark theme!
#or
import kplot.colorblind        #A style based around the perceptually uniform color map 'magma'
```

## Known Issues
Currently there are a couple of small quirks to kplot
  * There seems to be some inconsistency with the default plot size
  * You will have to restart your kernel in Jupyter to completely switch between styles, this is most notable when switching from the dark style.
