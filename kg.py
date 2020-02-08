import matplotlib.pyplot as plt
from glob import glob
import sys

def find():
    for p in sys.path:
        search = glob(p+'*/kplot/styles/kg.mplstyle')
        if len(search) == 1:
            return search[0]
           
s = find()
plt.style.use(s)