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

def double_osc_func_decay(x,amp,freq1,freq2,offset1,offset2,tau):
    return amp * np.cos(2 * np.pi * freq1 * (x - offset1))*np.cos(2 * np.pi * freq2 * (x - offset1))*np.exp(-(x - offset1)/tau) - offset2

f = Labber.LogFile('Z:\Projects\Fluxonium\Data\Augustus 18\\2019\\12\Data_1226\Ramsey_AWG_qubitB_2.hdf5')
# d = f.getEntry(0)
# for (channel, value) in d.items():
#     print(channel, ":", value)

time = f.getData('Multi-Qubit Pulse Generator - Sequence duration')[0]
signal = f.getData('AlazarTech Signal Demodulator - Channel A - Average demodulated value')*1e6
freq = f.getData('Multi-Qubit Pulse Generator - Frequency #1')[:,0]
axis_nice = np.linspace(time[0], time[-1], 1001)
freqFit = np.zeros((len(freq),2))
freqFit_err = np.zeros((len(freq),2))
plt.figure(1)
for freq_idx, f in enumerate(freq):
    toFit = np.real(signal[freq_idx,:])
    freq_guess1 = f - 136.3e6
    freq_guess2 = 0
    plt.plot(time*1e6, toFit)
    guess = ([np.max(toFit) - np.min(toFit), freq_guess1,freq_guess2, 0, toFit[0], 10e-6])
    opt, cov = curve_fit(double_osc_func_decay, ydata=toFit, xdata=time, p0=guess)
    err = np.sqrt(np.diag(cov))

    # plt.plot(axis_nice * 1e6, double_osc_func_decay(axis_nice, *opt))
    freqFit[freq_idx,0] = freq[freq_idx]-opt[1]
    freqFit[freq_idx, 1] = freq[freq_idx]-opt[2]
    freqFit_err[freq_idx, 0] = err[1]
    freqFit_err[freq_idx, 1] = err[2]

plt.figure(2)
plt.errorbar(x=freq/1e6,y=freqFit[:,0]/1e6,yerr=freqFit_err[:,0]/1e6)
# plt.errorbar(x=freq/1e6,y=freqFit[:,1]/1e6,yerr=freqFit_err[:,1]/1e6)
print (np.mean(freqFit[:,0]))
plt.show()