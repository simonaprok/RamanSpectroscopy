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
from matplotlib.ticker import AutoMinorLocator
import os
from os import listdir
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from ipywidgets import interact
import numpy.polynomial.polynomial as poly

# This file can be modified for whatever plotting purpose
# the name of the directory with data files
Load1= 'Load1_new' # name of the directory which to plot
Graphs = 'Load1Graphs' # name of directory to save the graphs
raw = 'Pristine_new' # name of with a pristine baseline-corrected data frame
# return current directory path
path = os.path.abspath(os.getcwd()) # get current working directory path
graphs = os.path.join(path, Graphs) # get the path to graphs
load1 = os.path.join(path, Load1) # get the path to the data
raw0 = os.path.join(path, raw) # get the path to the pristine data
pristine1 = sorted(os.listdir(raw0)) # sort the files in a directory
list= sorted(os.listdir(load1)) # sort the files in a directory






for framename in list: # in file names

    set_name = os.path.join(load1, framename) # find the path to file
    set_graphs_name = os.path.join(graphs, framename) # find the path to file
    df = pd.read_csv(set_name) # use this if this is a baseline-corrrected data otherwise put #
    #df = pd.read_csv(set_name, names=['Shift','Intensity']) # use this if this is an originally stored data
    intensities = np.array(df['Intensity']) # store intenisty data
    shift = np.array(df['Shift']) # store shift data
    plt.rc('font', family='serif', size=15)
    fig = plt.figure(figsize=(13.33,7.5))
    ax = fig.add_subplot(111)
    plt.tight_layout()
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
    def onpick3(event):  # click events
        ind = event.ind
        print('onpick3 scatter:', ind, shift[ind]) # print shift values

    plt.ticklabel_format(style='scientific', axis='y',scilimits=(0,0))
    plt.gca().tick_params(which='both', direction="in")
    plt.gca().xaxis.set_minor_locator(AutoMinorLocator())
    plt.gca().yaxis.set_minor_locator(AutoMinorLocator())
    plt.gca().xaxis.set_ticks_position('both')
    plt.gca().yaxis.set_ticks_position('both')
    ax.scatter(shift,intensities, picker=True, s=10)  # allows to click on the data points
    #ax.plot(shift0,intensities0/np.max(intensities0), lw=1)  # scaled if needed
    ax.plot(shift,intensities, lw=0.5)
    ax.set_xlim(left=100, right=np.max(shift))
    ax.set_ylim(bottom=-10000, top=np.max(intensities)+1000)
    ax.set_title(str(framename))
    ax.set_xlabel(r'Wavenumber shift / cm$^{-1}$ ')
    ax.set_ylabel('Intensity')
    fig.canvas.mpl_connect('pick_event', onpick3) # call the click events
    #plt.savefig(set_graphs_name+'.pdf', edgecolor='b', dpi=600, bbox_inches='tight')  # save to graph folder 
    plt.show()
