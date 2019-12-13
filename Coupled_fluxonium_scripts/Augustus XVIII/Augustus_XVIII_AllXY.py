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

# f = Labber.LogFile('C:\Data\Projects\Fluxonium\Data\Augustus 18\\2019\\11\Data_1120\Histogram.hdf5')
# guess1D = np.array([435, -480, 715, -305, 1300, -78, 1400, 86, 60])
f = Labber.LogFile('C:\Data\Projects\Fluxonium\Data\Augustus 18\\2019\\12\Data_1212\AllXY_heralded_qubit_A.hdf5')
# d = f.getEntry(0)
# for (channel, value) in d.items():
#     print(channel, ":", value)

signal = f.getData('AlazarTech Signal Demodulator - Channel A - Demodulated values')
freq = f.getData('IQ 1 - Frequency')[:21, 0]
amplitude = f.getData('Multi-Qubit Pulse Generator - Amplitude #1')[::21, 0]
# print (signal.shape)
# print (freq/1e9)
# print (amplitude)

xmin1 = -350
xmax1 = -100
ymin1 = -1000
ymax1 = -750
allxy_signal_preselected = np.zeros((11,21,21), dtype = complex)
for amplitude_index in range(len(amplitude)):
    for freq_index in range(len(freq)):
        for pulse_index in range(21):
            herald_signal = signal[amplitude_index*len(freq)*21 + freq_index*21 + pulse_index, 0::2]*1e6
            select_signal = signal[amplitude_index*len(freq)*21 + freq_index*21 + pulse_index, 1::2]*1e6
            sReal = np.real(herald_signal)
            sImag = np.imag(herald_signal)
            preselected_signal1 = []
            for idy in range(len(herald_signal)):
                if (sReal[idy] > xmin1) and (sReal[idy] < xmax1) and (sImag[idy] > ymin1) and (sImag[idy] < ymax1):
                    preselected_signal1 = np.append(preselected_signal1, select_signal[idy])
            allxy_signal_preselected[amplitude_index, freq_index, pulse_index] = np.average(preselected_signal1)
#
plt.figure(1)
for idy in range(11):
    for idx in range(21):
        plt.plot(np.real(allxy_signal_preselected[idy, idx, :]))

# plt.figure(2)
# plt.plot(np.imag(allxy_signal_preselected))
plt.show()