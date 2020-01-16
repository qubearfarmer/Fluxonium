import numpy as np
import sys
sys.path.append('C:\Program Files (x86)\Labber\Script')
import Labber
from matplotlib import pyplot as plt
from qutip import*
from scipy.optimize import curve_fit
from matplotlib import style

style.use('seaborn-paper')
#constants
kB = 1.38e-23
h = 6.626e-34

def osc_func_decay(x,amp,freq,offset1,offset2,tau):
    return amp * np.cos(2 * np.pi * freq * (x - offset1))*np.exp(-(x - offset1)/tau) - offset2
######################################
# f = Labber.LogFile('Z:\Projects\Fluxonium\Data\Augustus 18\\2019\\12\Data_1220\IQ_modulation_gaussian_width.hdf5')
#
# d = f.getEntry(0)
# for (channel, value) in d.items():
#     print(channel, ":", value)
# centerFreq = f.getData('Rohde&Schwarz Spectrum Analyzer - Center frequency')[0,0]
# span = f.getData('Rohde&Schwarz Spectrum Analyzer - Span')[0,0]
# num_pts = f.getData('Rohde&Schwarz Spectrum Analyzer - # of points')[0,0]
# signal = f.getData('Rohde&Schwarz Spectrum Analyzer - Signal')
# width = f.getData('Multi-Qubit Pulse Generator - Width')[0]
# freq = np.linspace(centerFreq - span/2, centerFreq + span/2, int(num_pts))*1e-9
# for idx in range(len(signal[:,0])):
#     plt.plot(freq, signal[idx,:], label = str(np.round(width[idx]*1e9,3))+'ns')
# plt.xlim([0.95,1.05])
# plt.xticks(np.linspace(0.95,1.05,5))
######################################
# f = Labber.LogFile('Z:\Projects\Fluxonium\Data\Augustus 18\\2019\\12\Data_1220\IQ_modulation_gaussian_width_80MHz.hdf5')

# d = f.getEntry(0)
# for (channel, value) in d.items():
#     print(channel, ":", value)
# centerFreq = f.getData('Rohde&Schwarz Spectrum Analyzer - Center frequency')[0,0]
# span = f.getData('Rohde&Schwarz Spectrum Analyzer - Span')[0,0]
# num_pts = f.getData('Rohde&Schwarz Spectrum Analyzer - # of points')[0,0]
# signal = f.getData('Rohde&Schwarz Spectrum Analyzer - Signal')
# width = f.getData('Multi-Qubit Pulse Generator - Width')[0]
# freq = np.linspace(centerFreq - span/2, centerFreq + span/2, int(num_pts))*1e-9
# for idx in range(len(signal[:,0])):
#     plt.plot(freq, signal[idx,:], label = str(np.round(width[idx]*1e9,3))+'ns')
# plt.xlim([0.95,1.05])
# plt.xticks(np.linspace(0.95,1.05,5))

#########################################################
#IQ modulation when carrier freq < 80MHz
# f = Labber.LogFile('Z:\Projects\Fluxonium\Data\Augustus 18\\2019\\12\Data_1220\IQ_modulation_gaussian_width_75MHz.hdf5')
# centerFreq = f.getData('Rohde&Schwarz Spectrum Analyzer - Center frequency')[0,0]
# span = f.getData('Rohde&Schwarz Spectrum Analyzer - Span')[0,0]
# num_pts = f.getData('Rohde&Schwarz Spectrum Analyzer - # of points')[0,0]
# signal = f.getData('Rohde&Schwarz Spectrum Analyzer - Signal')[0]
# freq = np.linspace(centerFreq - span/2, centerFreq + span/2, int(num_pts))*1e-9
# plt.plot(freq,signal)

#########################################################
#IQ modulation when carrier freq > 80MHz but the sideband is <80MHz
# f = Labber.LogFile('Z:\Projects\Fluxonium\Data\Augustus 18\\2019\\12\Data_1220\IQ_modulation_ACgaussian_width_75MHz.hdf5')
# centerFreq = f.getData('Rohde&Schwarz Spectrum Analyzer - Center frequency')[0,0]
# span = f.getData('Rohde&Schwarz Spectrum Analyzer - Span')[0,0]
# num_pts = f.getData('Rohde&Schwarz Spectrum Analyzer - # of points')[0,0]
# signal = f.getData('Rohde&Schwarz Spectrum Analyzer - Signal')[0]
# freq = np.linspace(0, span, int(num_pts))*1e-9
# plt.plot(freq,signal)

#IQ modulation when carrier freq > 80MHz, DC pulse
# f = Labber.LogFile('Z:\Projects\Fluxonium\Data\Augustus 18\\2019\\12\Data_1220\IQ_modulation_ACgaussian_width_135MHz.hdf5')
# centerFreq = f.getData('Rohde&Schwarz Spectrum Analyzer - Center frequency')[0,0]
# span = f.getData('Rohde&Schwarz Spectrum Analyzer - Span')[0,0]
# num_pts = f.getData('Rohde&Schwarz Spectrum Analyzer - # of points')[0,0]
# signal = f.getData('Rohde&Schwarz Spectrum Analyzer - Signal')[0]
# freq = np.linspace(0, span, int(num_pts))*1e-9
# plt.plot(freq,signal)

#IQ modulation when carrier freq > 80MHz, DC pulse
# f = Labber.LogFile('Z:\Projects\Fluxonium\Data\Augustus 18\\2019\\12\Data_1220\IQ_modulation_ACgaussian_width_75&135MHz.hdf5')
# centerFreq = f.getData('Rohde&Schwarz Spectrum Analyzer - Center frequency')[0,0]
# span = f.getData('Rohde&Schwarz Spectrum Analyzer - Span')[0,0]
# num_pts = f.getData('Rohde&Schwarz Spectrum Analyzer - # of points')[0,0]
# signal = f.getData('Rohde&Schwarz Spectrum Analyzer - Signal')[0]
# freq = np.linspace(0, span, int(num_pts))*1e-9
# plt.plot(freq,signal)

###################################################
#With AWG
# f = Labber.LogFile('Z:\Projects\Fluxonium\Data\Augustus 18\\2019\\12\\Data_1220\AWG_ACgaussian_width_75&135MHz.hdf5')
# centerFreq = f.getData('Rohde&Schwarz Spectrum Analyzer - Center frequency')[0,0]
# span = f.getData('Rohde&Schwarz Spectrum Analyzer - Span')[0,0]
# num_pts = f.getData('Rohde&Schwarz Spectrum Analyzer - # of points')[0,0]
# signal = f.getData('Rohde&Schwarz Spectrum Analyzer - Signal')[0]
# freq = np.linspace(0, span, int(num_pts))*1e-9
# plt.plot(freq,signal, label = 'Multiplexed pulse')
#
# f = Labber.LogFile('Z:\Projects\Fluxonium\Data\Augustus 18\\2019\\12\Data_1220\AWG_ACgaussian_width_75MHz.hdf5')
# centerFreq = f.getData('Rohde&Schwarz Spectrum Analyzer - Center frequency')[0,0]
# span = f.getData('Rohde&Schwarz Spectrum Analyzer - Span')[0,0]
# num_pts = f.getData('Rohde&Schwarz Spectrum Analyzer - # of points')[0,0]
# signal = f.getData('Rohde&Schwarz Spectrum Analyzer - Signal')[0]
# freq = np.linspace(0, span, int(num_pts))*1e-9
# plt.plot(freq,signal, label = '75MHz')
#
f = Labber.LogFile('Z:\Projects\Fluxonium\Data\Augustus 18\\2019\\12\Data_1220\AWG_ACgaussian_width_135MHz.hdf5')
centerFreq = f.getData('Rohde&Schwarz Spectrum Analyzer - Center frequency')[0,0]
span = f.getData('Rohde&Schwarz Spectrum Analyzer - Span')[0,0]
num_pts = f.getData('Rohde&Schwarz Spectrum Analyzer - # of points')[0,0]
signal = f.getData('Rohde&Schwarz Spectrum Analyzer - Signal')[0]
freq = np.linspace(0, span, int(num_pts))*1e-9
plt.plot(freq,signal, label = '135MHz')
#
# f = Labber.LogFile('Z:\Projects\Fluxonium\Data\Augustus 18\\2019\\12\Data_1221\AWG_ACgaussian_width_75&135MHz_separate_channels.hdf5')
# centerFreq = f.getData('Rohde&Schwarz Spectrum Analyzer - Center frequency')[0,0]
# span = f.getData('Rohde&Schwarz Spectrum Analyzer - Span')[0,0]
# num_pts = f.getData('Rohde&Schwarz Spectrum Analyzer - # of points')[0,0]
# signal = f.getData('Rohde&Schwarz Spectrum Analyzer - Signal')[0]
# freq = np.linspace(0, span, int(num_pts))*1e-9
# plt.plot(freq,signal, label = 'Local control')


plt.tick_params(labelsize = 16.0)
plt.legend(prop={'size': 12})
plt.ylabel('Power in (dBm)', size = 16.0)
plt.xlabel('Frequency (GHz)', size = 16.0)
plt.show()