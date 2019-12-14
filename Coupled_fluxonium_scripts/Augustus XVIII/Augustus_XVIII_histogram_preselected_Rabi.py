import numpy as np
import sys
sys.path.append('C:\Program Files (x86)\Labber\Script')
import Labber
from matplotlib import pyplot as plt
from qutip import*
from scipy.optimize import curve_fit

def gaussian4(x,a1,x1,a2,x2,a3,x3,a4,x4,sigma):
    return a1*np.exp(-(x-x1)**2/sigma**2) + a2*np.exp(-(x-x2)**2/sigma**2) + \
           a3*np.exp(-(x-x3)**2/sigma**2) + a4*np.exp(-(x-x4)**2/sigma**2)

def gaussian2d(coord,a,x0,y0,sigma):
    return a*np.exp((-(coord[0]-x0)**2-(coord[1]-y0)**2)/sigma**2)

def osc_func(x,amp,freq,offset1,offset2):
    return amp * np.cos(2 * np.pi * freq * (x - offset1)) - offset2

#constants
kB = 1.38e-23
h = 6.626e-34
############################################################
#Vary heralding wait time
f = Labber.LogFile('C:\Data\Projects\Fluxonium\Data\Augustus 18\\2019\\12\Data_1213\Rabi_heralded_qubit_B_I_2.hdf5')
# d = f.getEntry(0)
# for (channel, value) in d.items():
#     print(channel, ":", value)

signal = f.getData('AlazarTech Signal Demodulator - Channel A - Demodulated values')
pulseAmp = f.getData('Multi-Qubit Pulse Generator - Amplitude #1')[0]
rabi_signal = np.zeros(len(pulseAmp), dtype = complex)
rabi_signal_preselected_1 = np.zeros(len(pulseAmp), dtype = complex)
rabi_signal_preselected_2 = np.zeros(len(pulseAmp), dtype = complex)
rabi_signal_preselected_3 = np.zeros(len(pulseAmp), dtype = complex)
rabi_signal_preselected_4 = np.zeros(len(pulseAmp), dtype = complex)

xmin1 = -350
xmax1 = -100
ymin1 = -1000
ymax1 = -750

xmin2= 50
xmax2= 250
ymin2= -600
ymax2= -350

xmin3 = -650
xmax3 = -450
ymin3 = 10
ymax3 = 250

xmin4 = -900
xmax4 = -700
ymin4 = -250
ymax4 = 0
for idx in range(len(pulseAmp)):
    herald_signal = signal[idx,0::2]* 1e6
    select_signal = signal[idx,1::2]* 1e6
    sReal = np.real(herald_signal)
    sImag = np.imag(herald_signal)
    if idx == 0:
        H, xedges, yedges = np.histogram2d(sReal, sImag, bins=[100, 100])
        H = H.T
        X, Y = np.meshgrid(xedges, yedges)
        plt.pcolormesh(X, Y, H, cmap='GnBu')
        # plt.colorbar()
    rabi_signal[idx] = np.average(select_signal)
    preselected_signal1 = []
    preselected_signal2 = []
    preselected_signal3 = []
    preselected_signal4 = []
    for idy in range(len(herald_signal)):
        if (sReal[idy]>xmin1) and (sReal[idy]<xmax1) and (sImag[idy]>ymin1) and (sImag[idy]<ymax1):
            preselected_signal1 = np.append(preselected_signal1, select_signal[idy])
        elif (sReal[idy]>xmin2) and (sReal[idy]<xmax2) and (sImag[idy]>ymin2) and (sImag[idy]<ymax2):
            preselected_signal2 = np.append(preselected_signal2, select_signal[idy])
        elif (sReal[idy]>xmin3) and (sReal[idy]<xmax3) and (sImag[idy]>ymin3) and (sImag[idy]<ymax3):
            preselected_signal3 = np.append(preselected_signal3, select_signal[idy])
        elif (sReal[idy]>xmin4) and (sReal[idy]<xmax4) and (sImag[idy]>ymin4) and (sImag[idy]<ymax4):
            preselected_signal4 = np.append(preselected_signal4, select_signal[idy])

    rabi_signal_preselected_1[idx] = np.average(preselected_signal1)
    rabi_signal_preselected_2[idx] = np.average(preselected_signal2)
    rabi_signal_preselected_3[idx] = np.average(preselected_signal3)
    rabi_signal_preselected_4[idx] = np.average(preselected_signal4)

plt.plot(np.real(rabi_signal), np.imag(rabi_signal), label='raw Rabi')
plt.plot(np.real(rabi_signal_preselected_1), np.imag(rabi_signal_preselected_1), label = 'preselect gg')
plt.plot(np.real(rabi_signal_preselected_2), np.imag(rabi_signal_preselected_2), label = 'preselect eg')
plt.plot(np.real(rabi_signal_preselected_3), np.imag(rabi_signal_preselected_3), label = 'preselect ge')
plt.plot(np.real(rabi_signal_preselected_4), np.imag(rabi_signal_preselected_4), label = 'preselect ee')
plt.legend()
plt.xlabel('I (uV)')
plt.ylabel('Q (uV)')

plt.figure(2)
plt.plot(pulseAmp, np.real(rabi_signal_preselected_1))
freq_guess = 0.4
guess = ([np.max(np.real(rabi_signal_preselected_1)) - np.min(np.real(rabi_signal_preselected_1)),freq_guess,0,np.real(rabi_signal_preselected_1)[0]])
opt, cov = curve_fit(osc_func,ydata = np.real(rabi_signal_preselected_1), xdata = pulseAmp, p0=guess)
axis_nice = np.linspace(pulseAmp[0], pulseAmp[-1], 1001)
plt.plot(axis_nice, osc_func(axis_nice,*opt))

plt.show()


