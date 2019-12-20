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

def lorentzian(x, amp ,x0,y0, width):
    return amp*width/((x-x0)**2 + width**2)+y0
#####################################################################################
path = 'C:\Data\Projects\Fluxonium\Data\Augustus 18\\2019\\12\Data_1213\Two_tone_spec_IQ1_5.hdf5'
f = Labber.LogFile(path)
d = f.getEntry(0)
for (channel, value) in d.items():
    print(channel, ":", value)
# print ("Number of entries: ", f.getNumberOfEntries())

qubit_freq = f.getData('IQ 1 - Frequency')[0]*1e-6
signal = f.getData('AlazarTech Signal Demodulator - Channel A - Average demodulated value')[0]*1e6
signal = IQ_rotate(signal)
signal_real = np.real(signal)
signal_imag = np.imag(signal)
signal_mag = np.sqrt(signal_real**2 + signal_imag**2)
signal_real = signal_real - np.mean(signal_real)
fitSig = signal_mag

plt.plot(qubit_freq, fitSig)
plt.xlabel('Frequency (MHz)', size = 16.0)
plt.ylabel('I signal (uV)', size = 16.0)
plt.tick_params(labelsize = 16.0)
# # plt.colorbar()

#fit
guess = ([np.max(fitSig)-np.min(fitSig), 126.36, fitSig[0], 0.2])
opt,cov = curve_fit(lorentzian, xdata = qubit_freq, ydata = fitSig, p0=guess)

axis_nice = np.linspace(qubit_freq[0], qubit_freq[-1], 1001)
plt.plot(axis_nice, lorentzian(axis_nice, *opt))
plt.title(np.round(opt[1],3))
plt.show()