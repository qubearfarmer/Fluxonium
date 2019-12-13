import numpy as np
from matplotlib import pyplot as plt
import sys
from scipy.optimize import curve_fit
sys.path.append('C:\Program Files (x86)\Labber\Script')
import Labber

def osc_func(x,amp,freq,offset1,offset2):
    return amp * np.cos(2 * np.pi * freq * (x - offset1)) - offset2

def find_freq1(y_data, x_data):
    y = np.fft.fft((np.max(y_data) - np.min(y_data)) ** -1.0 * y_data)
    f = np.fft.fftfreq(len(x_data)) * (x_data[1] - x_data[0]) ** -1
    return abs(f[np.argmax(y)])

f = Labber.LogFile('G:\Projects\Fluxonium\Data\Augustus 18\\2019\\11\Data_1109\Rabi_B.hdf5')
# d = f.getEntry(0)
# for (channel, value) in d.items():
#     print(channel, ":", value)
signal= f.getData('AlazarTech Signal Demodulator - Channel A - Average demodulated value')[0]
pulse_width = f.getData('Multi-Qubit Pulse Generator - Width')[0]