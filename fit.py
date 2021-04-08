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


def baseline(y, y_new): # reassingment conditions for data corrected data
    base = []
    j = 0
    for i in range(0,len(y)):
        if y_new[i]>y[i]: # if the baseline intensities are bigger than the original
            base.append(y[i]) # append the original values
            j +=1
        else:
            base.append(y_new[i]) # otherwise  append the baseline results
    base = np.array(base) # make and array
    return base, j

def plot_fit(x, y, yfit, name, R): # plotting stuff
    plt.rc('font', family='serif')
    fig = plt.figure(figsize=(13.33,7.5))
    ax = fig.add_subplot(111)
    ax.plot(x,y, 'grey', label='data')
    ax.plot(x, yfit, 'k--', label=r'polynomial fit, R$^2$=' + str(R))
    ax.set_title(name)
    ax.set_xlabel(r'Wavenumber shift / cm$^{-1}$ ')
    ax.set_ylabel('Intensity')
    #ax.set_ylim(bottom=np.min(y), top=75000)
    ax.set_xlim(left=np.min(x), right=np.max(x))
    ax.legend(loc='upper left')
def plot_baseline(x, y, yfit, name): # baseline plotting stuff 
    fig = plt.figure(figsize=(13.33,7.5))
    ax = fig.add_subplot(111)
    ax.plot(x,y, 'grey', label='data')
    ax.plot(x, yfit, 'k-', label='baseline fit')
    ax.set_title(name)
    ax.set_xlabel(r'Wavenumber shift / cm$^{-1}$ ')
    ax.set_ylabel('Intensity')
    #ax.set_ylim(bottom=np.min(y), top=75000)
    ax.set_xlim(left=np.min(x), right=np.max(x))
    ax.legend(loc='upper left')
