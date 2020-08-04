import numpy as np
import sys
sys.path.append('C:\Program Files (x86)\Labber\Script')
import Labber
from matplotlib import pyplot as plt
from qutip import*
from scipy.optimize import curve_fit

#constants
kB = 1.38e-23
h = 6.626e-34

f = Labber.LogFile('C:\Data\Projects\Fluxonium\Data\Augustus 18\\2019\\11\Data_1120\Histogram.hdf5')
guess1D = np.array([435, -750, 715, -500, 1300, -100, 1200, 100, 60])
# f = Labber.LogFile('C:\Data\Projects\Fluxonium\Data\Augustus 18\\2019\\12\Data_1205\Histogram_heralding_check.hdf5')
# guess1D = np.array([1435, -800, 715, -605, 2300, -150, 2400, 86, 60])

def gaussian (x,a,x0,sigma):
    return a*np.exp(-(x-x0)**2/sigma**2)
def gaussian4(x,a1,x1,a2,x2,a3,x3,a4,x4,sigma):
    return a1*np.exp(-(x-x1)**2/sigma**2) + a2*np.exp(-(x-x2)**2/sigma**2) + \
           a3*np.exp(-(x-x3)**2/sigma**2) + a4*np.exp(-(x-x4)**2/sigma**2)
def gaussian2d(coord,a,x0,y0,sigma):
    return a*np.exp((-(coord[0]-x0)**2-(coord[1]-y0)**2)/sigma**2)

signal = f.getData('AlazarTech Signal Demodulator - Channel A - Demodulated values')
# delay_time = f.getData('Multi-Qubit Pulse Generator - Delay after heralding')[:,0]
repetition = 10
signal = signal[0]
sReal = np.real(signal)*1e6
sImag = np.imag(signal)*1e6
H, xedges, yedges = np.histogram2d(sReal,sImag, bins = [100,100])
H = H.T
##########################################################
#2D fit
plt.figure(1, figsize = [7,7])
X,Y = np.meshgrid(xedges, yedges)
plt.pcolormesh(X,Y,H, cmap = 'GnBu')
plt.xlim([-950,370])
plt.tick_params(labelsize = 18.0)
path = 'C:\\Users\\nguyen89\Google Drive\Research\Illustration\Thesis\Chapter 6 gates\\histogram1.pdf'
plt.savefig(path, dpi=300)
##########################################################
#1D fit
plt.figure(2, figsize = [8,7])
counts, edges = np.histogram(sReal, bins = 100)
plt.plot(edges[:-1], counts,'.')
opt,cov = curve_fit(gaussian4, edges[:-1],counts, guess1D)
axis_nice = np.linspace(edges[0], edges[-1], 1001)
plt.plot(axis_nice, gaussian4(axis_nice,*opt), linewidth = 2.0)
a1,x1,a2,x2,a3,x3,a4,x4,sigma = opt
plt.plot(axis_nice, gaussian(axis_nice,a1,x1,sigma), linestyle = '--', linewidth = 1.0)
plt.plot(axis_nice, gaussian(axis_nice,a2,x2,sigma), linestyle = '--', linewidth = 1.0)
plt.plot(axis_nice, gaussian(axis_nice,a3,x3,sigma), linestyle = '--', linewidth = 1.0)
plt.plot(axis_nice, gaussian(axis_nice,a4,x4,sigma), linestyle = '--', linewidth = 1.0)
pgg = a4/(a1+a2+a3+a4)
peg = a3/(a1+a2+a3+a4)
pge = a1/(a1+a2+a3+a4)
pee = a2/(a1+a2+a3+a4)
f_a = 72.4e6
f_b = 136.3e6
T_a = h*f_a/(-kB*np.log((pee+peg)/(pgg+pge)))
T_b = h*f_b/(-kB*np.log((pee+pge)/(pgg+peg)))
populations = dict({'P_gg=': round(pgg,3), 'P_eg=': round(peg,3), 'P_ge=': round(pge,3), 'P_ee=': round(pee,3)})
temperatures = dict({'T_A(mK)=': round(T_a*1e3,3), 'T_B(mK)=':round(T_b*1e3,3)})
textstr =""
for key in temperatures:
    textstr = textstr+key+str(temperatures[key])+'\n'
# plt.text(-1000,1000,textstr)
print (temperatures)
print (populations)
plt.tick_params(labelsize = 18.0)
plt.ylim([0,1500])
plt.xlim([-950,370])
path = 'C:\\Users\\nguyen89\Google Drive\Research\Illustration\Thesis\Chapter 6 gates\\histogram2.pdf'
plt.savefig(path, dpi=300)
plt.show()