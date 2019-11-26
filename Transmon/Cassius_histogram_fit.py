import numpy as np
from matplotlib import pyplot as plt
import sys
from scipy.optimize import curve_fit

sys.path.append('C:\Program Files (x86)\Labber\Script')
import Labber

##################################################################################
# One tone superposition, sweep cavity power
path = 'G:\Projects\Fluxonium\Data\Cassius I\\2019\\09\Data_0926\one_tone_hist_cavity_pow_sweep.hdf5'
f = Labber.LogFile(path)

signal = f.getData('Signal Demodulation - Value - Single shot')
hist, bin_edges = np.histogram(np.real(signal[0])*1e3,bins=100)
signal_I = bin_edges[0:-1]
plt.plot(signal_I,hist, '--')

def double_gaussian(x,amp1,std1,amp2,std2):
    mean1 = 1.731
    mean2 = 0.532
    return amp1*np.exp(-(x-mean1)**2/std1**2) + amp2*np.exp(-(x-mean2)**2/std2**2)
# def gaussian(x,amp,mean,std):
#     return amp*np.exp(-(x-mean)**2/std**2)
#
guess = ([7000, 0.2, 300, 0.2])
opt,cov = curve_fit(double_gaussian,ydata=hist, xdata=signal_I,p0=guess)
plt.plot(signal_I,double_gaussian(signal_I,*opt), color = 'red', linewidth = 2.0)
# plt.plot(signal_I,double_gaussian(signal_I,1,0.2,1,0.2), color = 'orange', linewidth = 2.0)
#
# guess = ([7000,1.731,0.3])
# opt,cov = curve_fit(gaussian,ydata=counts, xdata=x,p0=guess)
# # plt.plot(x,gaussian(x,*opt), linewidth = 2.0)
# guess = ([200,0.532,0.3])
# opt,cov = curve_fit(gaussian,ydata=counts, xdata=x,p0=guess)
# plt.plot(x,gaussian(x,*opt), linewidth = 2.0)

# opt,cov = curve_fit(double_gaussian,ydata=counts, xdata=x,p0=guess)
# plt.plot(x,double_gaussian(x,*opt), color = 'blue', linewidth = 2.0)

plt.show()