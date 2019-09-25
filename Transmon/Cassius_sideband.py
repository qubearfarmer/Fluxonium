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


path = 'G:\Projects\Fluxonium\Data\Cassius I\\2019\\09\Data_0917\Two_tone_sideband_4.hdf5'
f = Labber.LogFile(path)
# d = f.getEntry(0)
# for (channel, value) in d.items():
#     print(channel, ":", value)
# print ("Number of entries: ", f.getNumberOfEntries())

signal = f.getData('Signal Demodulation - Value')
qubit_freq = f.getData('Qubit RF - Frequency')[0]
pump_freq = f.getData('R&S IQ 2 - Frequency')[:,0]
for idx in range(len(pump_freq)-1):
    signal[idx,:] = IQ_rotate(signal[idx,:])

signal = signal[:-1,:]
signal_real = np.real(signal)
signal_real = signal_real - np.mean(signal_real)

X, Y = np.meshgrid(pump_freq/1e9, qubit_freq/1e9)
Z = abs(signal_real*1e6).transpose()

plt.pcolor(X,Y,Z,cmap = 'GnBu', vmin = 0, vmax = 100)
plt.xlabel('Pump tone frequency (GHz)', size = 16.0)
plt.ylabel('Qubit tone frequency (GHz)', size = 16.0)
plt.tick_params(labelsize = 16)

#fit
#bluesideband
# cavity_freq = 7.512e9
# sideband_freq = 6.0753e9+7.5167e9-pump_freq
# plt.plot(pump_freq/pump_freq1e9,sideband_freq/1e9, linewidth = 2.0, color = 'blue')


#With power

plt.show()