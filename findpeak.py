import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import plotly.graph_objects as go
#%matplotlib notebook
from scipy import spatial
from plotly.callbacks import Points, InputDeviceState
import sys
from scipy.spatial import KDTree
import math
import scipy.optimize as opt
from scipy.optimize import curve_fit
from scipy.optimize import minimize
from scipy.stats import norm
import scipy.stats as stats
import scipy.integrate as integrate
import scipy.special as special
from scipy.integrate import quad, dblquad
import matplotlib.colors as mcolors
from scipy.signal import find_peaks
import csv
import os
from os import listdir
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from ipywidgets import interact
import numpy.polynomial.polynomial as poly

def plot_peaks(x, y):
    for i, j in zip(x, y):
        ax.text(i, j+700, np.int(i), ha='center', rotation='vertical', fontsize ='small')

# the name of the directory with data files
Load1, set = 'Load1', np.arange(21,40,1)

# return current directory path
path = os.path.abspath(os.getcwd())
# current dirctory
base = os.path.basename(path)

load1 = os.path.join(path, Load1)

list1 = sorted(os.listdir(load1))

set_name = os.path.join(load1, list1[0])

df = pd.read_csv(set_name, names=['Shift', 'Intensity'])
intensities = np.array(df['Intensity'])
shift = np.array(df['Shift'])
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(shift,intensities, picker=True, s=10)
ax.plot(shift,intensities)
ax.set_title(str(list1[0]))
ax.set_xlabel(r'Wavenumber shift / cm$^{-1}$ ')
ax.set_ylabel('Intensity')
coords=[]
# Simple mouse click function to store coordinates
def onclick(event):
    global ix, iy
    ix, iy = event.xdata, event.ydata
    print('x = %d, y = %d'%(
        ix, iy))

    global coords
    coords.append((ix, iy))

    return coords

def onpick3(event):
        ind = event.ind
        print('onpick3 scatter:', ind, shift[ind], intensities[ind])
        coords.append(ind)
        return coords

fig.canvas.mpl_connect('pick_event', onpick3)
#cid = fig.canvas.mpl_connect('button_press_event', onclick)
plt.show()
coords = np.array(coords)
print(shift[coords])
print(intensities[coords])
xy = np.array([np.ravel(shift[coords]), np.ravel(intensities[coords])])
print(np.shape(xy))
#np.savetxt(str(list1[0])+'.txt', xy)
result = np.loadtxt(str(list1[0])+'.txt')

fig = plt.figure()
ax = fig.add_subplot(111)
#ax.scatter(shift,intensities, picker=True, s=10)
ax.plot(shift,intensities, 'grey')
ax.set_title(str(list1[0]))
ax.set_xlabel(r'Wavenumber shift / cm$^{-1}$ ')
ax.set_ylabel('Intensity')
ax.set_ylim(bottom=np.min(intensities), top=75000)
ax.set_xlim(left=np.min(shift), right=np.max(shift))
plot_peaks(result[0],result[1])
plt.savefig('peaks.pdf', edgecolor='b')
plt.show()
#np.savetxt("out.txt", np.array([shift[coords], intensities[coords]]), fmt="%s")

#print(xy[spatial.KDTree(xy, compact_nodes=True).query(coords, p=1)[1]])
#print(coords)


#ptx = np.array(np.where(shift, intensity == (find_nearest(shift, coords[0][0]))))
#pty = np.array(np.where(intensity == (find_nearest(shift, coords[0][0]))))
#print(shift[pt1])
