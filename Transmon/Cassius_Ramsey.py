import numpy as np
from matplotlib import pyplot as plt
import sys
from scipy.optimize import curve_fit
sys.path.append('C:\Program Files (x86)\Labber\Script')
import Labber

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

def osc_func_decay(x,amp,freq,offset1,offset2,t2):
    return amp * np.cos(2 * np.pi * freq * (x - offset1))*np.exp(-(x-offset1)/t2) - offset2

def exp_decay(x, amp, tau, offset1, offset2):
    return amp*np.exp(-(x-offset1)/tau) - offset2

def line(x,slope,offset):
    return slope*x - offset

def find_freq1(y_data, x_data):
    y = np.fft.fft((np.max(y_data) - np.min(y_data)) ** -1.0 * y_data)
    f = np.fft.fftfreq(len(x_data)) * (x_data[1] - x_data[0]) ** -1
    return abs(f[np.argmax(y)])

def find_freq2(y_data, x_data):
    period = abs(x_data[np.argmax(y_data)] - x_data[np.argmin(y_data)]) * 2
    freq_guess = period ** -1
    return freq_guess

'''
################################################################################
#Flowers Ramsey
path = 'G:\Projects\Fluxonium\Data\Cassius I\\2019\\09\Data_0910\Ramsey_flower.hdf5'
f = Labber.LogFile(path)
# d = f.getEntry(0)
# for (channel, value) in d.items():
#     print(channel, ":", value)
# print ("Number of entries: ", f.getNumberOfEntries())

time = f.getData('Multi-Qubit Pulse Generator - Sequence duration')[0]
qubit_freq = f.getData('Qubit RF - Frequency')[:,0]
signal = f.getData('AlazarTech Signal Demodulator - Channel A - Average demodulated value')
t2_array = []
t2_err_array = []
freq_array = []
freq_err_array = []
#Fit Ramsey
for idx in range(len(qubit_freq)):
    signal[idx,:] = IQ_rotate(signal[idx,:])
    demod_real = np.real(signal[idx,:])
    # plt.plot(time, demod_real)
    amplitude_guess = (np.max(demod_real) - np.min(demod_real))/2
    freq_guess = qubit_freq[idx] - 6.07533e9
    amplitude_offset = np.min(demod_real)
    try:
        t2_guess = 3e-6
        guess_list = ([-amplitude_guess, freq_guess, 0, amplitude_offset, t2_guess])
        popt,pcov = curve_fit(osc_func_decay, ydata = demod_real, xdata = time, p0=guess_list)
        perr = np.sqrt(np.diag(pcov))
    except RuntimeError:
        try:
            t2_guess = 1e-6
            guess_list = ([-amplitude_guess, freq_guess, 0, amplitude_offset, t2_guess])
            popt, pcov = curve_fit(osc_func_decay, ydata=demod_real, xdata=time, p0=guess_list)
            perr = np.sqrt(np.diag(pcov))
        except RuntimeError:
            print ('Cannot fit entry ', idx)
            continue
    if perr[-1] > t2_guess:
        continue
    t2_array = np.append(t2_array,popt[-1])
    freq_array = np.append(freq_array, popt[1])
    freq_err_array = np.append(freq_err_array, perr[1])
    t2_err_array = np.append(t2_err_array, perr[-1])

#Plot the flower Ramsey
plt.figure(1)
X,Y = np.meshgrid(time*1e6,qubit_freq/1e9)
Z = np.real(signal)
plt.pcolormesh(X,Y,Z)
plt.xlabel('Time (us)', size = 16.0)
plt.ylabel('RF frequency (GHz)', size = 16.0)
plt.tick_params(labelsize = 16.0)

#Plot the fitted values
plt.figure(2)
plt.errorbar(x=freq_array/1e6, y=t2_array*1e6, yerr=t2_err_array*1e6, xerr=freq_err_array/1e6, linestyle='none', marker='d', mfc='none', ecolor = 'green', mec='green', ms=5,mew=2)
plt.xlabel('Detuning (MHz)', size = 16.0)
plt.ylabel('T2 (us)', size = 16.0)
plt.tick_params(labelsize = 16.0)
plt.show()
'''
##################################################################
#Ramsey with cavity photons
path = 'G:\Projects\Fluxonium\Data\Cassius I\\2019\\09\Data_0914\Ramsey_cavity photons_3.hdf5'
f = Labber.LogFile(path)
d = f.getEntry(0)
for (channel, value) in d.items():
    print(channel, ":", value)
print ("Number of entries: ", f.getNumberOfEntries())

# time = f.getData('Multi-Qubit Pulse Generator - Sequence duration')[0]
# qubit_freq = f.getData('Qubit RF - Frequency')[:,0]
# signal = f.getData('AlazarTech Signal Demodulator - Channel A - Average demodulated value')