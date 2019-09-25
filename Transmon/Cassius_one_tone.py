import numpy as np
from matplotlib import pyplot as plt
import sys
from scipy.optimize import curve_fit
sys.path.append('C:\Program Files (x86)\Labber\Script')
import Labber

# path = 'G:\Projects\Fluxonium\Data\Cassius I\\2019\\09\Data_0913\One_tone_2.hdf5'
# f = Labber.LogFile(path)
# d = f.getEntry(0)
# for (channel, value) in d.items():
#     print(channel, ":", value)
# print ("Number of entries: ", f.getNumberOfEntries())
#
# cavity_freq = f.getData('Cavity RF - Frequency')[0]
# signal = f.getData('Signal Demodulation - Value')[0]
# signal_real = np.real(signal)
# signal_imag = np.imag(signal)
# signal_mag = np.sqrt(signal_real**2 + signal_imag**2)
#
# plt.plot(cavity_freq/1e9, signal_mag*1e6)
# plt.xlabel('Frequency (GHz)', size = 16.0)
# plt.ylabel('Signal magnitude (uV)', size = 16.0)
# plt.tick_params(labelsize = 16.0)

#
# ###################################################################################
# path = 'G:\Projects\Fluxonium\Data\Cassius I\\2019\\09\Data_0913\One_tone_pi_pulse.hdf5'
# f = Labber.LogFile(path)
# d = f.getEntry(0)
# for (channel, value) in d.items():
#     print(channel, ":", value)
# print ("Number of entries: ", f.getNumberOfEntries())

# cavity_freq = f.getData('Cavity RF - Frequency')[0]
# signal = f.getData('Signal Demodulation - Value')[0]
# signal_real = np.real(signal)
# signal_imag = np.imag(signal)
# signal_mag = np.sqrt(signal_real**2 + signal_imag**2)
# plt.plot(cavity_freq/1e9, signal_mag*1e6)
# ######################################################################################
# path = 'G:\Projects\Fluxonium\Data\Cassius I\\2019\\09\Data_0917\One_tone_TWPA_off_2.hdf5'
# f = Labber.LogFile(path)
# # d = f.getEntry(0)
# # for (channel, value) in d.items():
# #     print(channel, ":", value)
# # print ("Number of entries: ", f.getNumberOfEntries())
#
# cavity_freq = f.getData('Cavity RF - Frequency')[0]
# signal = f.getData('Signal Demodulation - Value')[0]
# signal_real = np.real(signal)
# signal_imag = np.imag(signal)
# signal_mag = np.sqrt(signal_real**2 + signal_imag**2)
# plt.plot(cavity_freq/1e9, signal_mag*1e6)
######################################################################################


#2D
path = 'G:\Projects\Fluxonium\Data\Cassius I\\2019\\09\Data_0916\One_tone.hdf5'
f = Labber.LogFile(path)
# d = f.getEntry(0)
# # for (channel, value) in d.items():
# #     print(channel, ":", value)
# # print ("Number of entries: ", f.getNumberOfEntries())
#
cavity_freq = f.getData('Cavity RF - Frequency')[0]
cavity_power = f.getData('Cavity RF - Power')[:,0]
signal = f.getData('Signal Demodulation - Value')
signal_real = np.real(signal)
signal_imag = np.imag(signal)
signal_mag = np.sqrt(signal_real**2 + signal_imag**2)
X,Y = np.meshgrid(cavity_freq/1e9, cavity_power)
Z = (signal_mag*1e3)
plt.figure(1)
plt.pcolormesh(X,Y,Z,cmap = 'GnBu', vmin = 0, vmax = 2)
plt.xlabel('Frequency (GHz)', size = 16.0)
plt.ylabel('Cavity power (dBm)', size = 16.0)
plt.xlim([7.4, 7.6])
plt.tick_params(labelsize = 16)
#
path = 'G:\Projects\Fluxonium\Data\Cassius I\\2019\\09\Data_0917\One_tone_pi_pulse.hdf5'
f = Labber.LogFile(path)
d = f.getEntry(0)
for (channel, value) in d.items():
    print(channel, ":", value)
print ("Number of entries: ", f.getNumberOfEntries())

cavity_freq = f.getData('Cavity RF - Frequency')[0]
cavity_power = f.getData('Cavity RF - Power')[:,0]
signal = f.getData('Signal Demodulation - Value')
signal_real = np.real(signal)
signal_imag = np.imag(signal)
signal_mag = np.sqrt(signal_real**2 + signal_imag**2)
X,Y = np.meshgrid(cavity_freq/1e9, cavity_power)
Z = (signal_mag*1e3)
plt.figure(2)
plt.pcolormesh(X,Y,Z,cmap = 'GnBu', vmin = 0, vmax = 2)
plt.xlabel('Frequency (GHz)', size = 16.0)
plt.ylabel('Cavity power (dBm)', size = 16.0)
plt.tick_params(labelsize = 16)

plt.show()
