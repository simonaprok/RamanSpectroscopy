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

result = np.loadtxt(str(list1[0])+'.txt')

#polynomial degree
deg = 9

df = pd.read_csv(set_name, names=['Shift', 'Intensity'])
intensities = np.array(df['Intensity'])
shift = np.array(df['Shift'])


# coefficients got from qtiplot polyfit aka initial guess
coefs = [7.4508517482198e+02,1.5113842893392e+00, 5.1398682603007e-03, 1.7506507687846e-06,
-4.0340023361544e-09,-1.6256840635350e-13, 1.2626312972139e-15,-4.0203393383366e-19, 3.7224285887598e-23]
#the iniital fit
ffit = poly.polyval(shift, coefs)
namefit1= '0 iteration: polynomial fit '+str(list1[0])
fit.plot_fit(shift,intensities, ffit, namefit1)
plt.savefig('poly.pdf', edgecolor='b')
plt.show()
ffit_new1 = fit.baseline(intensities, ffit)
namebase1 = '0 iteration: baseline correction '+str(list1[0])
fit.plot_baseline(shift,intensities, ffit_new1, namebase1)
plt.savefig('base.pdf', edgecolor='b')
plt.show()
coefs1 = poly.polyfit(shift, ffit_new1, deg)
ffit1 = poly.polyval(shift, coefs1)

namefit1= '1 iteration: polynomial fit '+str(list1[0])
fit.plot_fit(shift,intensities, ffit1, namefit1)
plt.savefig('poly1.pdf', edgecolor='b')
plt.show()
ffit_new2 = fit.baseline(intensities, ffit1)
namebase2 = '1 iteration: baseline correction '+str(list1[0])
fit.plot_baseline(shift,intensities, ffit_new2, namebase2)
plt.savefig('base1.pdf', edgecolor='b')
plt.show()
coefs2 = poly.polyfit(shift, ffit_new2, deg)
ffit2 = poly.polyval(shift, coefs2)

namefit2= '2 iteration: polynomial fit '+str(list1[0])
fit.plot_fit(shift,intensities, ffit2, namefit2)
plt.show()

ffit_new3 = fit.baseline(intensities, ffit2)
coefs3 = poly.polyfit(shift, ffit_new3, deg)
ffit3 = poly.polyval(shift, coefs3)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(shift,intensities)
ax.plot(shift,ffit_new3)

plt.show()

ffit_new4 = fit.baseline(intensities, ffit3)
coefs4 = poly.polyfit(shift, ffit_new4, deg)
ffit4 = poly.polyval(shift, coefs4)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(shift,intensities)
ax.plot(shift,ffit_new4)

plt.show()

ffit_new5 = fit.baseline(intensities, ffit4)
coefs5 = poly.polyfit(shift, ffit_new5, deg)
ffit5 = poly.polyval(shift, coefs5)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(shift,intensities)
ax.plot(shift,ffit_new5)
plot_peaks(result[0],result[1])
plt.show()

ffit_new6 = fit.baseline(intensities, ffit5)
coefs6 = poly.polyfit(shift, ffit_new6, deg)
ffit6 = poly.polyval(shift, coefs6)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(shift,intensities)
ax.plot(shift,ffit_new6)

plt.show()

ffit_new7 = fit.baseline(intensities, ffit6)
coefs7 = poly.polyfit(shift, ffit_new7, deg)
ffit7 = poly.polyval(shift, coefs7)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(shift,intensities)
ax.plot(shift,ffit_new7)

plt.show()

ffit_new8 = fit.baseline(intensities, ffit7)
coefs8 = poly.polyfit(shift, ffit_new8, deg)
ffit8 = poly.polyval(shift, coefs8)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(shift,intensities)
ax.plot(shift,ffit_new8)
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(shift,intensities-ffit_new8, 'grey')
ax.set_xlabel(r'Wavenumber shift / cm$^{-1}$ ')
ax.set_ylabel('Intensity')

ax.axhline(y=0, ls='--', c='black')
ax.set_xlim(left=np.min(shift), right=np.max(shift))
ax.set_title('Baseline corrected'+ str(list1[0]))
plt.savefig('difference.pdf', edgecolor='b')
#plot_peaks(result[0],result[1])
plt.show()
