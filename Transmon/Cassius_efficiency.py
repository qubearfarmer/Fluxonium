import numpy as np
from matplotlib import pyplot as plt
import sys
from scipy.optimize import curve_fit
sys.path.append('C:\Program Files (x86)\Labber\Script')
import Labber

path = 'G:\Projects\Fluxonium\Data\Cassius I\\2019\\09\Data_0920\One_tone_hist.hdf5'
f = Labber.LogFile(path)
# d = f.getEntry(0)
# for (channel, value) in d.items():
#     print(channel, ":", value)
# print ("Number of entries: ", f.getNumberOfEntries())

signal = f.getData('Signal Demodulation - Value - Single shot')[0]
signal_real = np.real(signal)
signal_imag = np.imag(signal)
signal_amp = np.mean(signal_real)**2+np.mean(signal_imag)**2
signal_noise = np.var(signal_real) + np.var(signal_imag)
efficiency = signal_amp/signal_noise * (25.5*3.5)**-1
print (efficiency*100)
# plt.show()