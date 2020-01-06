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

def osc_func_decay(x,amp,freq,offset1,offset2,tau):
    return amp * np.cos(2 * np.pi * freq * (x - offset1))*np.exp(-(x - offset1)/tau) - offset2

def double_osc_func_decay(x,amp, freq1, freq2,offset1,offset2,tau):
    return amp * np.cos(2 * np.pi * freq1 * (x - offset1))*np.cos(2 * np.pi * freq2 * (x - offset1))*np.exp(-(x - offset1)/tau) - offset2

f = Labber.LogFile('Z:\Projects\Fluxonium\Data\Augustus 18\\2019\\12\Data_1230\Ramsey_AWG_qubitA_2.hdf5')

# d = f.getEntry(0)
# for (channel, value) in d.items():
#     print(channel, ":", value)

time = f.getData('Multi-Qubit Pulse Generator - Sequence duration')[0]
signal = f.getData('AlazarTech Signal Demodulator - Channel A - Average demodulated value')[0]*1e6
toFit = np.abs(signal)
plt.plot(time*1e6, toFit)


freq_guess = 0.3e6
guess = ([np.max(toFit) - np.min(toFit),freq_guess,0,toFit[0],5e-6])
opt, cov = curve_fit(osc_func_decay,ydata = toFit, xdata = time, p0=guess)
axis_nice = np.linspace(time[0], time[-1], 1001)
plt.plot(axis_nice*1e6, osc_func_decay(axis_nice,*opt))
plt.xlabel('us')
plt.ylabel('I (uV)')
ramsey_freq = opt[1]/1e6 #MHz
ramsey_tau = opt[-1]*1e6 #us
title = str(round(ramsey_freq,3)) +'MHz, ' + str(round(ramsey_tau,3)) + 'us'
plt.title(title)

# freq1_guess = 0.4e6
# freq2_guess = 20e3
# guess = ([np.max(np.real(signal)) - np.min(np.real(signal)),freq1_guess,freq2_guess,0,np.real(signal)[0],15e-6])
# opt, cov = curve_fit(double_osc_func_decay,ydata = np.real(signal), xdata = time, p0=guess)
# axis_nice = np.linspace(time[0], time[-1], 1001)
# plt.plot(axis_nice*1e6, double_osc_func_decay(axis_nice,*opt))
# plt.xlabel('us')
# plt.ylabel('I (uV)')
# ramsey_freq1 = opt[1]/1e6 #MHz
# ramsey_freq2 = opt[2]/1e6 #MHz
# ramsey_tau = opt[-1]*1e6 #us
# title = str(round(ramsey_freq1,3))+', ' + str(round(ramsey_freq2,3)) +'MHz, ' + str(round(ramsey_tau,3)) + 'us'
# plt.title(title)

plt.show()