import numpy as np
import sys
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
from qutip import*
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
#######################################################################################
path = 'G:\Projects\Fluxonium\Data\Cassius I\\2019\\09\Data_0919\Qubit_tomography.hdf5'
f = Labber.LogFile(path)
# d = f.getEntry(0)
# for (channel, value) in d.items():
#     print(channel, ":", value)
# print ("Number of entries: ", f.getNumberOfEntries())

number_pulses = f.getData('Multi-Qubit Pulse Generator - # of pulses')[0]
pre_pulse = f.getData('Multi-Qubit Pulse Generator - Pulse')
post_pulse = f.getData('Multi-Qubit Pulse Generator - Tomography pulse index 1-QB')
signal = f.getData('Signal Demodulation - Value')
# print(number_pulses)
# print(pre_pulse)
# print(post_pulse)

#pulse sequence: 1/ nothing or pulse, 2/ X or X/2 3/ 3 post pulses
#Input
signal[:,0] = IQ_rotate(signal[:,0])
signal[:,1] = IQ_rotate(signal[:,1])
print(np.real(signal[:,0]))
print(np.real(signal[:,1]))
#signal0 = I-meas, I-meas, I-X/2pmeas, I-X/2pmeas, I-Y/2mmeas, I-Y/2mmeas
#signal1 = X-meas, X/2-meas, X-X/2pmeans, X/2p-X/2pmeas, X-Y/2mmeas, X/2p-Y/2mmeas

signal_gnd = np.mean([np.real(signal[0,0]),np.real(signal[1,0])])
signal_etd = np.real(signal[0,1])
pulse_num = 1
pre_pulse_index = 0 #0 for X or 1 for X/2
signal_I, signal_X2p, signal_Y2m = np.real(signal[pre_pulse_index::2,pulse_num])
sz,sy,sx = (np.real(signal[pre_pulse_index::2,pulse_num])-signal_etd)/(signal_gnd-signal_etd)*2-1



# rho = 0.5*(signal_gnd*qeye(2)+signal_Y2m*sigmax()+signal_X2p*sigmay()+signal_I*sigmaz())
rho = 0.5*(qeye(2)+sx*sigmax()+1j*sy*sigmay()+sz*sigmaz())
print (rho)
ket = basis(2,1)
# ket = ket/np.sqrt(2)
rho_ideal = ket2dm(ket)
fig, ax = matrix_histogram(rho,  ['0','1'],['0','1'], limits=[0,1])
ax.view_init(azim=-45, elev=45)

T=tracedist(rho, rho_ideal)
F = np.sqrt(1 - T**2)
print (F)


# plt.figure(2)
# x = np.linspace(0,3,4)
# y = abs(1+sz,sx-1j*sy, sx+1j*sy,1+sz)
# plt.bar(x,y)
plt.show()



