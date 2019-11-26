import numpy as np
from matplotlib import pyplot as plt
import sys
from scipy.optimize import curve_fit
sys.path.append('C:\Program Files (x86)\Labber\Script')
import Labber

##################################################################################
#One tone superposition, sweep cavity power
# path = 'G:\Projects\Fluxonium\Data\Cassius I\\2019\\09\Data_0914\One_tone_hist_6.hdf5'
# f = Labber.LogFile(path)

# signal = f.getData('Signal Demodulation - Value - Single shot')
# cavity_power = f.getData('Cavity RF - Power')[0]

# for idx in range(len(cavity_power)):
#     signal_real = np.real(signal[idx])
#     signal_imag = np.imag(signal[idx])
#     plt.hist2d(signal_real*1e3, signal_imag*1e3, bins = 100,range = [[-1, 2], [-1, 2]])
#     plt.title('Cavity power = '+ str(cavity_power[idx])+ 'dBm', size = 14.0)
#     plt.xlabel('I (mV)')
#     plt.ylabel('Q (mV)')
#     # Save it
#     filename = 'C:\\Users\\nguyen89\Documents\For animation\One_tone_hist_cavity_power_' + str(idx) + '.png'
#     plt.savefig(filename, dpi=96)
#     plt.gca()

##################################################################################
#Rabi histogram
# path = 'G:\Projects\Fluxonium\Data\Cassius I\\2019\\09\Data_0914\Rabi_hist_7.hdf5'
# f = Labber.LogFile(path)
#
# signal = f.getData('Signal Demodulation - Value - Single shot')
# width = f.getData('Multi-Qubit Pulse Generator - Width')[0]
#
# for idx in range(len(width)):
#     signal_real = np.real(signal[idx])
#     signal_imag = np.imag(signal[idx])
#     plt.hist2d(signal_real*1e3, signal_imag*1e3, bins = 100,range = [[-1, 2], [-1, 2]])
#     # plt.title('Gaussian width = '+ str(width[idx]*1e9)+ 'ns', size = 14.0)
#     plt.xlabel('I (mV)')
#     plt.ylabel('Q (mV)')
#     # Save it
#     filename = 'C:\\Users\\nguyen89\Documents\For animation\One_tone_hist_Rabi_' + str(idx) + '.png'
#     plt.savefig(filename, dpi=96)
#     plt.gca()

##################################################################################
#Cavity freq histogram
# path = 'G:\Projects\Fluxonium\Data\Cassius I\\2019\\09\Data_0914\One_tone_hist_5.hdf5'
# f = Labber.LogFile(path)
#
# signal = f.getData('Signal Demodulation - Value - Single shot')
# freq = f.getData('Cavity RF - Frequency')[0]
#
# for idx in range(len(freq)):
#     signal_real = np.real(signal[idx])
#     signal_imag = np.imag(signal[idx])
#     plt.hist2d(signal_real*1e3, signal_imag*1e3, bins = 100,range = [[-2, 2], [-2, 2]])
#     plt.title('Cavity frequency = '+ str(freq[idx]*1e-9)+ 'GHz', size = 14.0)
#     plt.xlabel('I (mV)')
#     plt.ylabel('Q (mV)')
#     # Save it
#     filename = 'C:\\Users\\nguyen89\Documents\For animation\One_tone_hist_cavity_freq_' + str(idx) + '.png'
#     plt.savefig(filename, dpi=96)
#     plt.gca()

##################################################################################
#Integration time histogram
# path = 'G:\Projects\Fluxonium\Data\Cassius I\\2019\\09\Data_0914\One_tone_hist_7.hdf5'
# f = Labber.LogFile(path)
#
# signal = f.getData('Signal Demodulation - Value - Single shot')
# sample_num = f.getData('Digitizer - Number of samples')[0]
#
# for idx in range(len(sample_num)):
#     signal_real = np.real(signal[idx])
#     signal_imag = np.imag(signal[idx])
#     plt.hist2d(signal_real*1e3, signal_imag*1e3, bins = 100,range = [[-2, 2], [-2, 2]])
#     plt.title('Integration time = '+ str(sample_num[idx]*2)+ 'ns', size = 14.0)
#     plt.xlabel('I (mV)')
#     plt.ylabel('Q (mV)')
#     # Save it
#     filename = 'C:\\Users\\nguyen89\Documents\For animation\One_tone_hist_int_time_' + str(idx) + '.png'
#     plt.savefig(filename, dpi=96)
#     plt.gca()

##################################################################################
#1-2 cavity freq histogram
# path = 'G:\Projects\Fluxonium\Data\Cassius I\\2019\\09\Data_0918\Histogram_7.hdf5'
# f = Labber.LogFile(path)
#
# signal = f.getData('Signal Demodulation - Value - Single shot')
# freq = f.getData('Cavity RF - Frequency')[0]
#
# for idx in range(len(freq)):
#     signal_real = np.real(signal[idx])
#     signal_imag = np.imag(signal[idx])
#     plt.hist2d(signal_real*1e3, signal_imag*1e3, bins = 100,range = [[-2, 2], [-2, 2]])
#     plt.title('Cavity frequency = '+ str(freq[idx]*1e-9)+ 'GHz', size = 14.0)
#     plt.xlabel('I (mV)')
#     plt.ylabel('Q (mV)')
#     # Save it
#     filename = 'C:\\Users\\nguyen89\Documents\For animation\One_tone_hist_12CavityFreq_' + str(idx) + '.png'
#     plt.savefig(filename, dpi=96)
#     plt.gca()

##################################################################################
#1-2 Rabi histogram
path = 'G:\Projects\Fluxonium\Data\Cassius I\\2019\\09\Data_0918\Rabi_12_hist.hdf5'
f = Labber.LogFile(path)

signal = f.getData('Signal Demodulation - Value - Single shot')
width = f.getData('Multi-Qubit Pulse Generator - Amplitude #2')[0]

for idx in range(len(width)):
    signal_real = np.real(signal[idx])
    signal_imag = np.imag(signal[idx])
    plt.hist2d(signal_real*1e3, signal_imag*1e3, bins = 100,range = [[-2, 2], [-2, 2]])
    # plt.title('Gaussian width = '+ str(width[idx]*1e9)+ 'ns', size = 14.0)
    plt.xlabel('I (mV)')
    plt.ylabel('Q (mV)')
    # Save it
    filename = 'C:\\Users\\nguyen89\Documents\For animation\One_tone_hist_12Rabi_' + str(idx) + '.png'
    plt.savefig(filename, dpi=96)
    plt.gca()
##################################################################################
#1-2 T1 histogram
# path = 'G:\Projects\Fluxonium\Data\Cassius I\\2019\\09\Data_0918\T1(12)_hist.hdf5'
# f = Labber.LogFile(path)
#
# signal = f.getData('Signal Demodulation - Value - Single shot')
# width = f.getData('Multi-Qubit Pulse Generator - Sequence duration')[0]
#
# for idx in range(len(width)):
#     signal_real = np.real(signal[idx])
#     signal_imag = np.imag(signal[idx])
#     plt.hist2d(signal_real*1e3, signal_imag*1e3, bins = 100,range = [[-2, 2], [-2, 2]])
#     plt.title('Delay time = '+ str(width[idx]*1e6)+ 'us', size = 14.0)
#     plt.xlabel('I (mV)')
#     plt.ylabel('Q (mV)')
#     # Save it
#     filename = 'C:\\Users\\nguyen89\Documents\For animation\One_tone_hist_12T1_' + str(idx) + '.png'
#     plt.savefig(filename, dpi=96)
#     plt.gca()