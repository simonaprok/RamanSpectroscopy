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
# function for plotting peaks
def plot_peaks(x, y):
    for i, j in zip(x, y):
        ax.text(i, j+700, np.int(i), ha='center', rotation='vertical', fontsize ='small')

# the name of the directory with data files
Load1 = 'Load1_new' # name of the directory for which to assign the peaks
Peaks1 = 'PeaksLoad1' # name of the dirctory where the Raman bands data is stored
Graphs = 'Load1Graphs' # name of directory to save the graphs
# return current directory path
path = os.path.abspath(os.getcwd())
# current directory
base = os.path.basename(path)
n = 0 # choose the data frame index in Load 1
load1 = os.path.join(path, Load1) # path to data set
peaks1 = os.path.join(path, Peaks1)  # path to peak data
graphs1 = os.path.join(path, Graphs) # path to graphs
list1= sorted(os.listdir(load1)) # sort the data frames names alphabetically

framename = list1[n] # the data frame name in Load 1
set_name = os.path.join(load1, framename) #path to the data frame

set_peaks_name = os.path.join(peaks1, framename) # set the path to output peak data
set_graphs_name = os.path.join(graphs1, framename) # set the path to output graphs


# read the data frame
df = pd.read_csv(set_name) # put # if this is baseline-corrected data frame
#df = pd.read_csv(set_name, names=['Shift', 'Intensity']) # remove # if this is raw data frame
intensities = np.array(df['Intensity']) # array of intensities from the data frame
shift = np.array(df['Shift']) # array of wavenumber shifts from the data frame
""" # remove if this is a new peak assignment
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(shift,intensities, picker=True, s=20) # scatter plot intensities vs shift
ax.plot(shift,intensities) # plot intensities vs shift
ax.set_title(str(framename))
ax.set_xlabel(r'Wavenumber shift / cm$^{-1}$ ')
ax.set_ylabel('Intensity')
#plt.show()

coords=[] # coordinates are appended to here
# Simple mouse click function to store coordinates

def onpick3(event):
    ind = event.ind # index of a click event
    print('onpick3 scatter:', ind, shift[ind], intensities[ind])
    coords.append(ind) # append the coordinates
    return coords

fig.canvas.mpl_connect('pick_event', onpick3) # pick events

plt.show()
coords = np.array(coords) # make an array
print(shift[coords])  # print all shift values
print(intensities[coords]) # print all intensity value
xy = np.array([np.ravel(shift[coords]), np.ravel(intensities[coords])]) # ravel to save
print(np.shape(xy)) # print the shape

np.savetxt(set_peaks_name+'.txt', xy) # save the output
"""
result = np.loadtxt(set_peaks_name+'.txt') #load the peak  data
# plot of the graph with the assigned peaks
plt.rc('font', family='serif')
fig = plt.figure(figsize=(13.33,7.5))
ax = fig.add_subplot(111)
#ax.scatter(shift,intensities, picker=True, s=10)
ax.plot(shift,intensities, 'grey')
ax.set_title(str(framename))
ax.set_xlabel(r'Wavenumber shift / cm$^{-1}$ ')
ax.set_ylabel('Intensity')
ax.set_ylim(bottom=np.min(intensities), top=np.max(intensities)+10000)
ax.set_xlim(left=np.min(shift), right=np.max(shift))
plot_peaks(result[0],result[1]) # include the assigned peaks
#plt.savefig(set_graphs_name+'.pdf', edgecolor='b', dpi=600, bbox_inches='tight') # remove if want to save
plt.show()
