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

def osc_func(x,amp,freq,offset1,offset2):
    return amp * np.cos(2 * np.pi * freq * (x - offset1)) - offset2

def find_freq1(y_data, x_data):
    y = np.fft.fft((np.max(y_data) - np.min(y_data)) ** -1.0 * y_data)
    f = np.fft.fftfreq(len(x_data)) * (x_data[1] - x_data[0]) ** -1
    return abs(f[np.argmax(y)])

#####################################################################################
# path = 'G:\Projects\Fluxonium\Data\Cassius I\\2019\\09\Data_0913\Rabi_13.hdf5'
# f = Labber.LogFile(path)
# d = f.getEntry(0)
# for (channel, value) in d.items():
#     print(channel, ":", value)
# print ("Number of entries: ", f.getNumberOfEntries())
#
# pulse_width = f.getData('Multi-Qubit Pulse Generator - Width')[0]
# signal = f.getData('Signal Demodulation - Value')[0]
# signal = IQ_rotate(signal)
# signal_real = np.real(signal)
# signal_imag = np.imag(signal)

# plt.plot(pulse_width*1e9, signal_real*1e6)
# plt.xlabel('Pulse_width (ns)', size = 16.0)
# plt.ylabel('I signal (uV)', size = 16.0)
# plt.tick_params(labelsize = 16.0)
#####################################################################################
# path = 'G:\Projects\Fluxonium\Data\Cassius I\\2019\\09\Data_0913\Rabi_16.hdf5'
# f = Labber.LogFile(path)
# d = f.getEntry(0)
# for (channel, value) in d.items():
#     print(channel, ":", value)
# print ("Number of entries: ", f.getNumberOfEntries())
#
# pulse_width = f.getData('Multi-Qubit Pulse Generator - Width')[0]
# signal = f.getData('Signal Demodulation - Value')[0]
# signal = IQ_rotate(signal)
# signal_real = np.real(signal)
# signal_imag = np.imag(signal)
#
# plt.plot(pulse_width*1e9, signal_real*1e6)
# plt.xlabel('Pulse_width (ns)', size = 16.0)
# plt.ylabel('I signal (uV)', size = 16.0)
# plt.tick_params(labelsize = 16.0)

######################################################################################
path = 'G:\Projects\Fluxonium\Data\Cassius I\\2019\\09\Data_0914\Rabi_power_sweep.hdf5'
f = Labber.LogFile(path)
# d = f.getEntry(0)
# for (channel, value) in d.items():
#     print(channel, ":", value)
# print ("Number of entries: ", f.getNumberOfEntries())

pulse_width = f.getData('Multi-Qubit Pulse Generator - Width')[0]
cavity_power = f.getData('Cavity RF - Power')[:,0]
signal = f.getData('Signal Demodulation - Value')
osc_amplitude = np.zeros(len(cavity_power))
for idx in range(len(cavity_power)):
    signal[idx,:] = IQ_rotate(signal[idx,:])
    signal_real = np.real(signal[idx,:])
    signal_real = signal_real - np.mean(signal_real)
    signal_imag = np.imag(signal[idx,:])
    plt.figure(1)
    plt.plot(pulse_width*1e9, signal_real*1e6,'-d')
    plt.xlabel('Pulse_width (ns)', size = 16.0)
    plt.ylabel('I signal (uV)', size = 16.0)
    plt.tick_params(labelsize = 16.0)
    freq_guess = find_freq1(y_data =signal_real, x_data = pulse_width)
    guess = ([np.max(signal_real) - np.min(signal_real),freq_guess,0,0])
    opt, cov = curve_fit(osc_func,ydata = signal_real, xdata = pulse_width, p0=guess)
    plt.plot(pulse_width*1e9, osc_func(pulse_width,*opt)*1e6)
    osc_amplitude[idx] = opt[0]

plt.figure(2)
plt.plot(cavity_power, osc_amplitude*1e6)
plt.xlabel('Cavity Power (dBm)', size = 16.0)
plt.ylabel('Oscillation amplitude (uV)', size = 16.0)
plt.tick_params(labelsize = 16.0)

plt.show()