import numpy as np
from matplotlib import pyplot as plt
import sys
from scipy.optimize import curve_fit
sys.path.append('C:\Program Files (x86)\Labber\Script')
import Labber

def line(x,slope,offset):
    return slope*x - offset

def IQ_rotate(signal):
    #receives a signal and rotate it to real axis.
    demod_real = np.real(signal)
    demod_imag = np.imag(signal)
    guess_line = [(np.max(demod_imag)-np.min(demod_imag))/(np.max(demod_real)-np.min(demod_real)), np.min(demod_imag)]
    opt,cov = curve_fit(line,xdata=demod_real,ydata=demod_imag, p0 = guess_line)
    theta = np.arctan(opt[0])
    demod_real_rotate = demod_real*np.cos(theta) + demod_imag*np.sin(theta)
    demod_imag_rotate = -demod_real * np.sin(theta) + demod_imag * np.cos(theta)
    return demod_real_rotate + 1j*demod_imag_rotate

#############################################################################################
path = 'G:\Projects\Fluxonium\Data\Cassius I\\2019\\09\Data_0916\Pulse_tuneup_X2p.hdf5'
f = Labber.LogFile(path)
# d = f.getEntry(0)
# for (channel, value) in d.items():
#     print(channel, ":", value)
# print ("Number of entries: ", f.getNumberOfEntries())

signal = f.getData('Signal Demodulation - Value')
num_pulses = f.getData('Multi-Qubit Pulse Generator - # of pulses')[0]
pulse_amplitude = f.getData('Multi-Qubit Pulse Generator - Amplitude #1')[:,0]
variation = np.zeros(pulse_amplitude.shape)
for idx in range(len(pulse_amplitude)):
    signal[idx,:] = IQ_rotate(signal[idx,:])
    signal_real = np.real(signal[idx,:])
    signal_real = signal_real - np.mean(signal_real)
    variation[idx] = np.var(np.real(signal[idx,:]))
    # plt.plot(num_pulses,signal_real)


plt.plot(pulse_amplitude, variation)
plt.show()