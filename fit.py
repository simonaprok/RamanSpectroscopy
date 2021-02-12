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


def baseline(y, y_new):
    base = []
    for i in range(0,len(y)):
        if y_new[i]>y[i]:
            base.append(y[i])
        else:
            base.append(y_new[i])
    base = np.array(base)
    return base

def plot_fit(x, y, yfit, name):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x,y, 'grey', label='data')
    ax.plot(x, yfit, 'k--', label='polynomial fit')
    ax.set_title(name)
    ax.set_xlabel(r'Wavenumber shift / cm$^{-1}$ ')
    ax.set_ylabel('Intensity')
    #ax.set_ylim(bottom=np.min(y), top=75000)
    ax.set_xlim(left=np.min(x), right=np.max(x))
    ax.legend(loc='upper left')
def plot_baseline(x, y, yfit, name):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x,y, 'grey', label='data')
    ax.plot(x, yfit, 'k-', label='baseline fit')
    ax.set_title(name)
    ax.set_xlabel(r'Wavenumber shift / cm$^{-1}$ ')
    ax.set_ylabel('Intensity')
    #ax.set_ylim(bottom=np.min(y), top=75000)
    ax.set_xlim(left=np.min(x), right=np.max(x))
    ax.legend(loc='upper left')
