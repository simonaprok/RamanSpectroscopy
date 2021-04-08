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
from matplotlib.ticker import AutoMinorLocator
from os import listdir
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from ipywidgets import interact
import numpy.polynomial.polynomial as poly
from sklearn.metrics import r2_score

def plot_peaks(x, y): # function for plotting peaks
    for i, j in zip(x, y):
        ax.text(i, j+700, np.int(i), ha='center', rotation='vertical', fontsize ='small')
Load1 = 'Load1_new' # baseline-corrected data directory name
raw = 'Pristine_new' # baseline-corrected pristine data directory name
path = os.path.abspath(os.getcwd())
# current dirctory
base = os.path.basename(path)

load1 = os.path.join(path, Load1)
list1 = sorted(os.listdir(load1))
raw = os.path.join(path, raw)
pristine1 = sorted(os.listdir(raw))
list1 = [list1[i] for i in range(len(list1)) if i!=10 and i!=11 and i!=12 and i!=18]

plt.rc('font', family='serif', size=15)
#fig = plt.figure(figsize=(16.33,9.5),  facecolor='w', edgecolor='k')
fig = plt.figure(figsize=(10,16.5),  facecolor='w', edgecolor='k')
ax = fig.add_subplot(111)
plt.tight_layout()
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
#colormap = plt.cm.brg
#plt.gca().set_prop_cycle(plt.cycler('color', plt.cm.jet(np.linspace(0, 1, 20))))
plt.ticklabel_format(style='scientific', axis='y',scilimits=(0,0))
plt.gca().tick_params(which='both', direction="in")
plt.gca().xaxis.set_minor_locator(AutoMinorLocator())
plt.gca().yaxis.set_minor_locator(AutoMinorLocator())
plt.gca().xaxis.set_ticks_position('both')
plt.gca().yaxis.set_ticks_position('both')
set_name = os.path.join(raw, pristine1[0])
df = pd.read_csv(set_name)
intensities = np.array(df['Intensity'])
y = []
for j in intensities:
        if j > 15000:
            y.append(15000)
        else:
            y.append(j)
y = np.array(y)
shift = np.array(df['Shift'])
ax.plot(shift,y, lw=0.5, c='black')
#plt.savefig('stackplot.pdf', edgecolor='b')
ax.set_xlim(left=0, right=np.max(shift))
ax.set_xlabel(r'Wavenumber shift / cm$^{-1}$ ')
ax.set_ylabel('Intensity')
# assigned colour orange=Stage1, purple=Stage2, dodgerblue=Stage 3
colours = ['tab:orange', 'tab:orange', 'tab:orange','tab:orange', 'tab:purple','tab:purple','tab:purple','tab:orange','tab:orange','tab:purple','tab:orange','tab:purple', 'dodgerblue','dodgerblue','dodgerblue','dodgerblue']
offset = 5000
i = 15000
c=0
for framename in list1:

    set_name = os.path.join(load1, framename)
    df = pd.read_csv(set_name)
    intensities = np.array(df['Intensity'])
    intensities += i
    y = []
    for j in intensities:
            if j > 8000+i:
                y.append(8000+i)
            else:
                y.append(j)
    y = np.array(y)

    shift = np.array(df['Shift'])

    ax.plot(shift, y , lw=0.5, c=colours[c])
    ax.set_xlim(left=0, right=np.max(shift))
    ax.set_xlabel(r'Wavenumber shift / cm$^{-1}$ ')
    ax.set_ylabel('Intensity /a.u.')
    i += offset
    c +=1
ax.set_ylim(bottom=-10000, top=100000)
plt.vlines(x=[1450, 1650,2300,2800, 2950, 3350], ymin=-10000, ymax=i+4000, linestyles='dashed', colors = 'black', linewidths=0.5, )
plt.vlines(x=[930, 1050, 1250, 1410, 1440,1530, 1580, 1984, 2010, 2035, 2550, 2650 , 2755, 2840, 3100], ymin=0, ymax=i+4000, linestyles='dotted', colors = 'black', linewidths=0.5, )
#plt.savefig('stackplotUPD.pdf', edgecolor='b')
plt.close()

#classified data frame indices based on stages
stage1=[0,1,2,3,7,8,10]
stage2=[4,5,6,9,11]
stage3=[12,13,14,15]
stage = np.array([0,1,8,10,2,7,3,11,9,4,6,5,12,13,14,15])
list1 = np.array(list1)
list = list1[stage]
print(list)
plt.rc('font', family='serif', size=40)

fig = plt.figure(figsize=(10,14),  facecolor='w', edgecolor='k')
ax = fig.add_subplot(111)
plt.tight_layout()
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
#colormap = plt.cm.brg
#plt.gca().set_prop_cycle(plt.cycler('color', plt.cm.jet(np.linspace(0, 1, 20))))
plt.ticklabel_format(style='scientific', axis='y',scilimits=(0,0))
plt.gca().tick_params(which='both', direction="in")
plt.gca().xaxis.set_minor_locator(AutoMinorLocator())
plt.gca().yaxis.set_minor_locator(AutoMinorLocator())
plt.gca().xaxis.set_ticks_position('both')
plt.gca().yaxis.set_ticks_position('both')


set_name = os.path.join(raw, pristine1[0])
df1 = pd.read_csv(set_name)
#plt.savefig('stackplot.pdf', edgecolor='b')
intensities1 = np.array(df1['Intensity'])
y1 = []
for j in intensities1:
        if j > 15000:
            y1.append(15000)
        else:
            y1.append(j)
y1 = np.array(y1)
shift1 = np.array(df1['Shift'])
ax.plot(shift1,y1, lw=1, c='dimgrey')
#plt.savefig('stackplot.pdf', edgecolor='b')
ax.set_xlim(left=0, right=np.max(shift1))

colours1 = ['tab:orange', 'tab:orange', 'tab:orange','tab:orange', 'tab:orange','tab:orange','tab:orange','tab:purple','tab:purple','tab:purple','tab:purple','tab:purple', 'dodgerblue','dodgerblue','dodgerblue','dodgerblue','dodgerblue']
offset = 5000
k = 15000
l=0
for framename in list:

    #ax.set_title('Load1 stackplot')

    set_name = os.path.join(load1, framename)
    print(set_name)
    df = pd.read_csv(set_name)
    intensities = np.array(df['Intensity'])
    intensities += k
    y = []
    for j in intensities:
            if j > 8000+k:
                y.append(8000+k)
            else:
                y.append(j)
    y = np.array(y)

    shift = np.array(df['Shift'])

    ax.plot(shift, y , lw=1, c=colours1[l])
    ax.set_xlim(left=0, right=np.max(shift))
    ax.set_xlabel(r'Wavenumber shift / cm$^{-1}$ ')
    ax.set_ylabel('Intensity /a.u.')
    k += offset
    l +=1
print(l)
ax.set_ylim(bottom=-10000, top=100000)
plt.vlines(x=[1410], ymin=-10000, ymax=i+5000, linestyles='dotted', colors = 'dodgerblue', linewidths=1.5, label='stage3') #diamond C-C sp3
plt.vlines(x=[1440], ymin=-10000, ymax=i+5000, linestyles='dotted', colors = 'tab:purple', linewidths=1.5, ) # intermediate graphite-diamond sp2-sp3 C-C
plt.vlines(x=[1530], ymin=-10000, ymax=i+5000, linestyles='dotted', colors = 'dimgrey', linewidths=1.5, ) # C-H bending
plt.vlines(x=[1580], ymin=-10000, ymax=i+5000, linestyles='dotted', colors = 'tab:orange', linewidths=1.5, ) # graphitic C-C sp2
#plt.vlines(x=[1630], ymin=-10000, ymax=i+5000, linestyles='dotted', colors = 'tab:orange', linewidths=1.5, ) # С=С
plt.vlines(x=[1984], ymin=-10000, ymax=i+5000, linestyles='dotted', colors = 'tab:orange', linewidths=1.5, ) # C=O stretch
plt.vlines(x=[2010], ymin=-10000, ymax=i+5000, linestyles='dotted', colors = 'tab:orange', linewidths=1.5, ) # CC triple bond stretch
plt.vlines(x=[2035], ymin=-10000, ymax=i+5000, linestyles='dotted', colors = 'tab:purple', linewidths=1.5, ) # also CC triple bond stretch ?
plt.vlines(x=[2035], ymin=-10000, ymax=i+5000, linestyles='dotted', colors = 'tab:purple', linewidths=1.5, ) # CC triple bond stretch ?
plt.vlines(x=[2755], ymin=-10000, ymax=i+5000, linestyles='dotted', colors = 'tab:orange', linewidths=1.5, ) # defects ?
plt.vlines(x=[2840], ymin=-10000, ymax=i+5000, linestyles='dotted', colors = 'tab:orange', linewidths=1.5, ) # defects ?
plt.vlines(x=[2650], ymin=-10000, ymax=i+5000, linestyles='dotted', colors = 'dimgrey', linewidths=1.5, ) # defects ?


#plt.vlines(x=[1450, 1700,2150,2820, 2950, 3350], ymin=-10000, ymax=i+5000, linestyles='dashed', colors = 'dimgrey', linewidths=1, ) #pristine stuff
plt.vlines(x=[930, 1050, 1250, 2550,2650, 3100], ymin=-10000, ymax=i+5000, linestyles='dotted', colors = 'dimgrey', linewidths=1.5, )#pristine stuff
#plt.savefig('stackplotUPDcopy1.pdf', edgecolor='b', dpi=600)
#plt.legend(['stage3','stage2'],loc='lower left')
plt.show()
