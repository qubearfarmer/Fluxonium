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

#####################################################################################
path = 'G:\Projects\Fluxonium\Data\Cassius I\\2019\\09\Data_0913\Two_tone_2.hdf5'
f = Labber.LogFile(path)
d = f.getEntry(0)
for (channel, value) in d.items():
    print(channel, ":", value)
print ("Number of entries: ", f.getNumberOfEntries())

qubit_freq = f.getData('Qubit RF - Frequency')[0]
signal = f.getData('Signal Demodulation - Value')[0]
signal = IQ_rotate(signal)
signal_real = np.real(signal)
signal_imag = np.imag(signal)
signal_mag = np.sqrt(signal_real**2 + signal_imag**2)
# signal_real = signal_real - np.mean(signal_real)

plt.plot(qubit_freq*1e-9, signal_real*1e6)
plt.xlabel('Frequency (GHz)', size = 16.0)
plt.ylabel('I signal (uV)', size = 16.0)
plt.tick_params(labelsize = 16.0)
# plt.colorbar()
plt.show()