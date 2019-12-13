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

def osc_func(x,amp,freq,offset1,offset2):
    return amp * np.cos(2 * np.pi * freq * (x - offset1)) - offset2

f = Labber.LogFile('C:\Data\Projects\Fluxonium\Data\Augustus 18\\2019\\12\Data_1212\Ramsey_heralded_qubit_A.hdf5')
# d = f.getEntry(0)
# for (channel, value) in d.items():
#     print(channel, ":", value)

signal = f.getData('AlazarTech Signal Demodulator - Channel A - Demodulated values')
freq = f.getData('IQ 1 - Frequency')[:,0]
delayTime = f.getData('Multi-Qubit Pulse Generator - Sequence duration')[0]

freq_idx = 4
rabi_signal_preselected_1 = np.zeros(len(delayTime), dtype = complex)

xmin1 = -350
xmax1 = -100
ymin1 = -1000
ymax1 = -750

for time_idx in range(len(delayTime)):
    herald_signal = signal[freq_idx*len(delayTime)+time_idx, 0::2]* 1e6
    select_signal = signal[freq_idx*len(delayTime)+time_idx, 1::2]* 1e6
    sReal = np.real(herald_signal)
    sImag = np.imag(herald_signal)
    preselected_signal1 = []
    for idy in range(len(herald_signal)):
        if (sReal[idy]>xmin1) and (sReal[idy]<xmax1) and (sImag[idy]>ymin1) and (sImag[idy]<ymax1):
            preselected_signal1 = np.append(preselected_signal1, select_signal[idy])
    rabi_signal_preselected_1[time_idx] = np.average(preselected_signal1)

plt.figure(1)
plt.plot(delayTime*1e6, np.imag(rabi_signal_preselected_1))
plt.title(freq[freq_idx]*1e-6)

plt.figure(2)
plt.plot(delayTime*1e6, np.real(rabi_signal_preselected_1))
freq_guess = 0.5e6
guess = ([np.max(np.real(rabi_signal_preselected_1)) - np.min(np.real(rabi_signal_preselected_1)),freq_guess,0,np.real(rabi_signal_preselected_1)[0]])
opt, cov = curve_fit(osc_func,ydata = np.real(rabi_signal_preselected_1), xdata = delayTime, p0=guess)
axis_nice = np.linspace(delayTime[0], delayTime[-1], 1001)
plt.plot(axis_nice*1e6, osc_func(axis_nice,*opt))
plt.title(freq[freq_idx]*1e-6 - opt[1]/1e6)

plt.show()