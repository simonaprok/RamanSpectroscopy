import fit
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
from scipy.interpolate import interp1d
import csv
import os
from os import listdir
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from ipywidgets import interact
import numpy.polynomial.polynomial as poly
from sklearn.metrics import r2_score
from numpy import diff
from matplotlib.ticker import AutoMinorLocator

def pressure(v, A, B): # pressure gauge calculations
    v0 = 1334 #cm−1 # diamond edge at ambient conditions
    P = A*(v-v0)*(1+(B-1)*(v-v0)/(2*v0))/v0
    return P

K0 =  547 #GPa Bulk modulus
K0_dash = 3.75 # Bulk modulus pressure derivative

Load1 = 'Load1_new' # baseline-corrected data directory name

path = os.path.abspath(os.getcwd())
# current dirctory
base = os.path.basename(path)

load1 = os.path.join(path, Load1)
list1 = sorted(os.listdir(load1))
list1 = [list1[i] for i in range(len(list1)) if i!=10 and i!=11 and i!=12] # exclude noisy frames
for framename in list1:
    print(framename)
    set_name = os.path.join(load1, framename)
    df = pd.read_csv(set_name)


    plt.rc('font', family='serif')
    fig = plt.figure(figsize=(16.33,7.5))
    ax = fig.add_subplot(111)
    plt.ticklabel_format(style='scientific', axis='y',scilimits=(0,0))
    plt.gca().tick_params(which='both', direction="in")
    plt.gca().xaxis.set_minor_locator(AutoMinorLocator())
    plt.gca().yaxis.set_minor_locator(AutoMinorLocator())
    plt.gca().xaxis.set_ticks_position('both')
    plt.gca().yaxis.set_ticks_position('both')

    intensities = np.array(df['Intensity'])
    shift = np.array(df['Shift'])
    intensities_new = intensities[np.where((shift>1350) & (shift<1500))]
    shift_new = shift[np.where((shift>1350) & (shift<1500))]
    f = interp1d(shift_new , intensities_new, axis=0, fill_value='extrapolate' ) # interpolated data
    xnew = np.linspace(1349, 1450, num=3000)
    f_new = f(xnew)
    dydx = np.gradient(intensities_new)/np.gradient(shift_new) # find gradient
    v_min = shift_new[np.where(dydx==np.min(dydx))] # find gradient minimum in shift
    print('Minimum of the gradient (cm^-1):', v_min)
    def onpick3(event): # click events
        ind = event.ind # event index
        print('onpick3 scatter:', ind, shift_new[ind], intensities_new[ind]) # print corresponding shift, intenisty values
        half_intensity = (intensities_new[ind]+np.min(intensities_new))/2 # find 1/2 of the intensity between max and min in original data
        print('Diamond edge intensity:', half_intensity) # diamond edge intensity from original data estimates
        array = []
        for i in f_new: # in inerpolated intenisty
            array.append(np.abs(i-np.int(half_intensity))) # append absolute difference between interpolated intensity and 1/2 the intensity estimate

        edge =  xnew[np.where(array == np.min(array))] # estimate of diamond edge shift from interpolation
        print('Diamond edge location from maximum/minmium intensity estimations (cm^-1):', edge)
        closest1 = min(shift, key=lambda x:abs(x-edge)) #the closest neighbour to the edge in the original array
        if edge>closest1: # if the edge shift value is bigger than the closest data point
            index = np.where(shift ==closest1)[0] # find the index
            closest2 = shift[index+1]  # find the next nearest neighbour
        else: # otherwise
            index = np.where(shift==closest1)[0] # find the index
            closest2 = shift[index-1] # find the previous nearest neighbour
        print(closest1, closest2) # nearest neighbours
        err = abs(closest1-closest2)/2  # error in the diamond edge wavenumber shift estate
        print('Uncertainty in the shift position:', err)
        print('Pressure from maximum/minmum intensity estimations (GPa):', float(pressure(edge, K0,K0_dash)))


    print("Pressure from gradient (GPa):", float(pressure(v_min,K0,K0_dash)))
    print('Pick the diamond peak data point')
    ax.plot(shift_new,dydx, lw=1) # plot the gradient
    ax.scatter(shift_new,intensities_new, s=30, picker=True) # plot the original data
    ax.plot(shift_new,intensities_new, xnew, f(xnew),lw=1) # plot interpolated data
    ax.set_title(str(framename))
    ax.set_xlabel(r'Wavenumber shift / cm$^{-1}$ ')
    ax.set_ylabel('Intensity')
    ax.set_xlim(left=shift_new[0],right=shift_new[np.where(intensities_new==np.min(intensities_new))]) # limit at minimum wavenumber shift
    fig.canvas.mpl_connect('pick_event', onpick3) # pick by clicking
    plt.show()
