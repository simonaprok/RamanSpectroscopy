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
from sklearn.metrics import r2_score
from matplotlib.ticker import AutoMinorLocator

def plot_peaks(x, y): # function for plotting peaks
    for i, j in zip(x, y):
        ax.text(i, j+1000, np.int(i), ha='center', rotation='vertical', fontsize ='small')
def find_new_fit(x,y,y_fit, deg): # function for fitting a new baseline after every iteration
    ffit_new, count = fit.baseline(y, y_fit) # conditions to check which returns a reassigned baseline and number of points that were reassigned
    #fit.plot_baseline(x,y, ffit_new, namebase)
    #plt.savefig('base1.pdf', edgecolor='b')
    plt.show()
    coefs = poly.polyfit(x, ffit_new, deg) # coefficients after the least squares fit to the reassigned baseline
    #ffit = poly.polyval(x, coefs)
    #coefs = poly.Polynomial.fit(x, ffit_new, deg).convert().coef
    ffit = poly.polyval(x, coefs) # fit values
    #fit.plot_fit(x,y, ffit, namefit)
    #plt.show()

    return ffit, count, coefs

deg = 8 # polynomial degree to fit
index = 0 # index for the dataset in the folder assigned as Load1

# coefficients got from qtiplot polyfit aka initial guess
coefs = [7.450841068613885e+02,
1.511384834141590e+00,
5.139874053585957e-03,
1.750645632049827e-06,
-4.034004857898532e-09,
-1.625638841398842e-13,
1.262629286967346e-15,
-4.020335531914376e-19,
3.722425914659112e-23]

# the name of the directory with data files
Load1 = 'Load1' # raw data directory
Load1_new = 'Load1_new' # baseline-corrected directory
Graphs = 'Load1Graphs' # graph directory
Peaks1 = 'PeaksLoad1'  # peaks directory
# return current directory path
path = os.path.abspath(os.getcwd())
# current directory
base = os.path.basename(path)

load1 = os.path.join(path, Load1)
load1_new = os.path.join(path, Load1_new)
graphs1 = os.path.join(path, Graphs)
peaks1 = os.path.join(path, Peaks1)
peaks_list = sorted(os.listdir(peaks1))
list1 = sorted(os.listdir(load1))

framename = list1[index]
framename_corrected = framename + '_corrected'
framename_final = framename + '_final'
set_name = os.path.join(load1, framename)
set_name_new = os.path.join(load1_new, framename)
set_graphs_name_corrected = os.path.join(graphs1, framename_corrected)
set_graphs_name_final = os.path.join(graphs1, framename_final)
set_peaks_name = os.path.join(peaks1, framename)
result = np.loadtxt(set_peaks_name+'.txt')

print('Filename :', framename)
#polynomial degree

df = pd.read_csv(set_name, names=['Shift', 'Intensity'])
intensities = np.array(df['Intensity'])
shift = np.array(df['Shift'])

plt.rc('font', family='serif', size=15)
ffit = poly.polyval(shift, coefs)
namefit1= '0 iteration: polynomial fit '+str(framename)
Rstat = round(r2_score(intensities, ffit),4)
fit.plot_fit(shift,intensities, ffit, namefit1,Rstat)
#plt.savefig('poly.pdf', edgecolor='b')  # remove '#' if you want to save the initial graph with a fitted polynomial
plt.close()
#plt.show() # remove '#' if you want to see the intermediate step

print('Old R^2 :',r2_score(intensities, ffit))
ffit_new1, count1 = fit.baseline(intensities, ffit)
namebase1 = '0 iteration: baseline correction '+str(framename)
fit.plot_baseline(shift,intensities, ffit_new1, namebase1)
#plt.savefig('base.pdf', edgecolor='b')  # remove '#' if you want to save the graph with a reassigned baseline
plt.close() # put '#' if you want to see the intermediate step
#plt.show() # remove '#' if you want to see the intermediate step

coefs1 = poly.polyfit(shift, ffit_new1, deg)
#coefs1 = poly.Polynomial.fit(shift, ffit_new1, deg).convert().coef
ffit1 = poly.polyval(shift, coefs1)
Rstat1 = round(r2_score(intensities, ffit1),4)
namefit1= '1 iteration: polynomial fit '+str(framename)
fit.plot_fit(shift,intensities, ffit1, namefit1, Rstat1)
#plt.savefig(set_graphs_name+str(), edgecolor='b')  # remove '#' if you want to save the final itteration graph
plt.close() # put '#' if you want to see the intermediate step
#plt.show() # remove '#' if you want to see the intermediate step


number = 1
iteration = 0
finalfit = []
coefs_new = []

while number !=0: # while the number of points reassigned is not zero
    ffit_new_new, n,_ = find_new_fit(shift, intensities, ffit1, deg) # find the new fit
    ffit1 = ffit_new_new  # find the new fit
    number = n # number of points reassigned
    iteration = iteration+1 # count the new itteration
    if iteration==500: # if number of itteration is 500 break the loop
        ffit_new_new, n, coefs_new = find_new_fit(shift, intensities, ffit1, deg)
        finalfit = ffit_new_new
        coefs_new =coefs_new

        break
    if number == 0: # if number of points to reassign is 0 break the loop
        ffit_new_new, n, coefs_new = find_new_fit(shift, intensities, ffit1, deg)
        finalfit = ffit_new_new
        coefs_new =coefs_new

        break

print('New polynomial degree:', deg)
print('New polynomial coefficients: ', coefs_new )
print('The number of points not re-assigned:', number)
namefitfinal= str(iteration)+' iteration: polynomial fit '+str(framename)

print('R^2 :',r2_score(intensities, finalfit)) # R squared score
Rstatfin = round(r2_score(intensities, finalfit),4)
fit.plot_fit(shift,intensities, finalfit, namefitfinal, Rstatfin)
#plt.savefig(set_graphs_name_final+'.pdf', edgecolor='b') # remove '#' if you want to save the final itteration graph
print(np.shape(shift))
plt.show()

#count nonzero elements
print(np.count_nonzero(intensities-finalfit))

fig = plt.figure(figsize=(13.33,7.5)) # scale
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
plot_peaks(result[0],result[1]) # assigned peaks
ax.plot(shift,intensities-finalfit, 'grey',lw=1, c='red',label='baseline-corrected spectrum') # baseline-corrected data
ax.set_xlabel(r'Wavenumber shift / cm$^{-1}$ ')
ax.set_ylabel('Intensity/ a.u.')

ax.set_xlim(left=np.min(shift), right=np.max(shift))
#plt.savefig(set_graphs_name_corrected+'.pdf', edgecolor='b') # remove '#' if you want to save the baseline-corrected graph

# create a dataframe storing the values for the baseline-corrected data
df_new = pd.DataFrame({'Shift': shift, 'Intensity':intensities-finalfit}, index=None)
# save it to the new folder
df_new.to_csv(set_name_new,index=False)

ax.plot(shift,intensities+10000, 'grey', label='offset raw spectrum',lw=1, c='dodgerblue') # plot the original data
ax.plot(shift, finalfit+10000, 'k--', label=r'polynomial baseline',lw=1) # plot the fit
ax.set_title(name)
ax.set_ylim(bottom=np.min(0), top=80000) # limit in y
#ax.set_xlim(left=np.min(x), right=np.max(x)) # limit in x
ax.legend(loc='upper left')
#plt.savefig(set_graphs_name_corrected+'combined'+'.pdf', edgecolor='b') # save to graph folder if needed
plt.show()
