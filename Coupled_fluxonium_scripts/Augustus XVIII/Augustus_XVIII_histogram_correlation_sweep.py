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

#constants
kB = 1.38e-23
h = 6.626e-34
############################################################
#Vary heralding wait time
f = Labber.LogFile('C:\Data\Projects\Fluxonium\Data\Augustus 18\\2019\\12\Data_1205\Histogram_heralding_sweep.hdf5')
# d = f.getEntry(0)
# for (channel, value) in d.items():
#     print(channel, ":", value)

# signal = f.getData('AlazarTech Signal Demodulator - Channel A - Demodulated values')
# time = f.getData('Multi-Qubit Pulse Generator - Delay after heralding')[:,0]
# repetition = 10
# corr_array = np.zeros(len(time))
# T_A1 = np.zeros(len(time))
# T_B1 = np.zeros(len(time))
# T_A2 = np.zeros(len(time))
# T_B2 = np.zeros(len(time))
# for idy in range(len(time)):
#     signal_avg = []
#     for idx in range(repetition ):
#         signal_avg = np.append(signal_avg, signal[idy*repetition+idx, :])
#     herald_signal = signal_avg[0::2]
#     select_signal = signal_avg[1::2]
#     cov = np.corrcoef(herald_signal, select_signal)
#     corr_array[idy] = abs(cov[0,1])
#
#     sReal = np.real(herald_signal) * 1e6
#     sImag = np.imag(herald_signal) * 1e6
#     counts, edges = np.histogram(sReal, bins=100)
#     guess = np.array([3000, -800, 10000, -532, 13000, -200, 15000, 150, 60])
#     opt, cov = curve_fit(gaussian4, edges[:-1], counts, guess)
#     a1, x1, a2, x2, a3, x3, a4, x4, sigma = opt
#     pgg = a3 / (a1 + a2 + a3 + a4)
#     peg = a4 / (a1 + a2 + a3 + a4)
#     pge = a2 / (a1 + a2 + a3 + a4)
#     pee = a1 / (a1 + a2 + a3 + a4)
#     f_a = 72.5e6
#     f_b = 136.3e6
#     T_A1[idy] = h * f_a / (-kB * np.log((pee + peg) / (pgg + pge)))
#     T_B1[idy] = h * f_b / (-kB * np.log((pee + pge) / (pgg + peg)))
#
#     sReal = np.real(select_signal) * 1e6
#     sImag = np.imag(select_signal) * 1e6
#     counts, edges = np.histogram(sReal, bins=100)
#     opt, cov = curve_fit(gaussian4, edges[:-1], counts, guess)
#     a1, x1, a2, x2, a3, x3, a4, x4, sigma = opt
#     pgg = a3 / (a1 + a2 + a3 + a4)
#     peg = a4 / (a1 + a2 + a3 + a4)
#     pge = a2 / (a1 + a2 + a3 + a4)
#     pee = a1 / (a1 + a2 + a3 + a4)
#     T_A2[idy] = h * f_a / (-kB * np.log((pee + peg) / (pgg + pge)))
#     T_B2[idy] = h * f_b / (-kB * np.log((pee + pge) / (pgg + peg)))
#
# plt.figure(1)
# plt.plot(time*1e6, corr_array, linewidth = 2.0)
# plt.xlabel('Delay time (us)')
# plt.ylabel('Correlation between readout (max = 1)')
# plt.xlim([0,10])
# plt.ylim([0.77, 0.83])
# plt.figure(2)
# plt.plot(time*1e6, T_A1*1e3, linewidth = 2.0, label = 'First pulse qubit A')
# plt.plot(time*1e6, T_B1*1e3, linewidth = 2.0, label = 'First pulse qubit B')
# plt.plot(time*1e6, T_A2*1e3, linestyle = '--', linewidth = 2.0, label = 'Second pulse qubit A')
# plt.plot(time*1e6, T_B2*1e3, linestyle = '--', linewidth = 2.0, label = 'Second pulse qubit B')
# plt.legend()
# plt.xlabel('Delay time (us)')
# plt.ylabel('Temperatures (mK)')
############################################################################################
#Measure variation in 24h
f = Labber.LogFile('C:\Data\Projects\Fluxonium\Data\Augustus 18\\2019\\12\Data_1206\Histogram_correlation_repeat.hdf5')

# signal = f.getData('AlazarTech Signal Demodulator - Channel A - Demodulated values')
# repetition = len(signal[:,0])
# corr_array = np.zeros(repetition)
# T_A1 = np.zeros(repetition)
# T_B1 = np.zeros(repetition)
# T_A2 = np.zeros(repetition)
# T_B2 = np.zeros(repetition)
#
# for idy in range(repetition):
#     signal_avg = signal[idy,:]
#     herald_signal = signal_avg[0::2]
#     select_signal = signal_avg[1::2]
#     cov = np.corrcoef(herald_signal, select_signal)
#     corr_array[idy] = abs(cov[0,1])
#
#     sReal = np.real(herald_signal) * 1e6
#     sImag = np.imag(herald_signal) * 1e6
#     counts, edges = np.histogram(sReal, bins=100)
#     guess = np.array([3000, -800, 10000, -532, 13000, -200, 15000, 150, 60])
#     opt, cov = curve_fit(gaussian4, edges[:-1], counts, guess)
#     a1, x1, a2, x2, a3, x3, a4, x4, sigma = opt
#     pgg = a3 / (a1 + a2 + a3 + a4)
#     peg = a4 / (a1 + a2 + a3 + a4)
#     pge = a2 / (a1 + a2 + a3 + a4)
#     pee = a1 / (a1 + a2 + a3 + a4)
#     f_a = 72.5e6
#     f_b = 136.3e6
#     T_A1[idy] = h * f_a / (-kB * np.log((pee + peg) / (pgg + pge)))
#     T_B1[idy] = h * f_b / (-kB * np.log((pee + pge) / (pgg + peg)))
#
#     sReal = np.real(select_signal) * 1e6
#     sImag = np.imag(select_signal) * 1e6
#     counts, edges = np.histogram(sReal, bins=100)
#     opt, cov = curve_fit(gaussian4, edges[:-1], counts, guess)
#     a1, x1, a2, x2, a3, x3, a4, x4, sigma = opt
#     pgg = a3 / (a1 + a2 + a3 + a4)
#     peg = a4 / (a1 + a2 + a3 + a4)
#     pge = a2 / (a1 + a2 + a3 + a4)
#     pee = a1 / (a1 + a2 + a3 + a4)
#     T_A2[idy] = h * f_a / (-kB * np.log((pee + peg) / (pgg + pge)))
#     T_B2[idy] = h * f_b / (-kB * np.log((pee + pge) / (pgg + peg)))
#
# plt.figure(1)
# plt.plot(corr_array, linewidth = 2.0)
# plt.xlabel('Attempt')
# plt.ylabel('Correlation between readout (max = 1)')
# # plt.xlim([0,10])
# # plt.ylim([0.77, 0.83])
# plt.figure(2)
# plt.plot(T_A1*1e3, linewidth = 2.0, label = 'First pulse qubit A')
# plt.plot(T_B1*1e3, linewidth = 2.0, label = 'First pulse qubit B')
# plt.plot(T_A2*1e3, linestyle = '--', linewidth = 2.0, label = 'Second pulse qubit A')
# plt.plot(T_B2*1e3, linestyle = '--', linewidth = 2.0, label = 'Second pulse qubit B')
# plt.legend()
# plt.xlabel('Attempt')
# plt.ylabel('Temperatures (mK)')
############################################################################################
f = Labber.LogFile('C:\Data\Projects\Fluxonium\Data\Augustus 18\\2019\\12\Data_1206\Histogram_correlation_v_cavity_power.hdf5')
# d = f.getEntry(0)
# for (channel, value) in d.items():
#     print(channel, ":", value)
signal = f.getData('AlazarTech Signal Demodulator - Channel A - Demodulated values')
power = f.getData('Cavity RF - Power')[0]

# repetition = 10
# corr_array = np.zeros(len(power))
# T_A1 = np.zeros(len(power))
# T_B1 = np.zeros(len(power))
# T_A2 = np.zeros(len(power))
# T_B2 = np.zeros(len(power))
# # #
# for idy in range(len(power)):
#     signal_avg = signal[idy,:]
#     herald_signal = signal_avg[0::2]
#     select_signal = signal_avg[1::2]
#     cov = np.corrcoef(herald_signal, select_signal)
#     corr_array[idy] = abs(cov[0,1])
#
#     sReal = np.real(herald_signal) * 1e6
#     sImag = np.imag(herald_signal) * 1e6
#     counts, edges = np.histogram(sReal, bins=100)
#     # if idy <
#     guess = np.array([500, -550, 800, -332, 1000, -80, 1200, 120, 30])
#     opt, cov = curve_fit(gaussian4, edges[:-1], counts, guess)
#     a1, x1, a2, x2, a3, x3, a4, x4, sigma = opt
#     pgg = a3 / (a1 + a2 + a3 + a4)
#     peg = a4 / (a1 + a2 + a3 + a4)
#     pge = a2 / (a1 + a2 + a3 + a4)
#     pee = a1 / (a1 + a2 + a3 + a4)
#     f_a = 72.5e6
#     f_b = 136.3e6
#     T_A1[idy] = h * f_a / (-kB * np.log((pee + peg) / (pgg + pge)))
#     T_B1[idy] = h * f_b / (-kB * np.log((pee + pge) / (pgg + peg)))
#
#     sReal = np.real(select_signal) * 1e6
#     sImag = np.imag(select_signal) * 1e6
#     counts, edges = np.histogram(sReal, bins=100)
#     opt, cov = curve_fit(gaussian4, edges[:-1], counts, guess)
#     a1, x1, a2, x2, a3, x3, a4, x4, sigma = opt
#     pgg = a3 / (a1 + a2 + a3 + a4)
#     peg = a4 / (a1 + a2 + a3 + a4)
#     pge = a2 / (a1 + a2 + a3 + a4)
#     pee = a1 / (a1 + a2 + a3 + a4)
#     T_A2[idy] = h * f_a / (-kB * np.log((pee + peg) / (pgg + pge)))
#     T_B2[idy] = h * f_b / (-kB * np.log((pee + pge) / (pgg + peg)))
# #
# plt.figure(1)
# plt.plot(power, corr_array, linewidth = 2.0)
# plt.xlabel('Power (dBm)')
# plt.ylabel('Correlation between readout (max = 1)')
# # plt.xlim([0,10])
# # plt.ylim([0.77, 0.83])
# plt.figure(2)
# plt.plot(power, T_A1*1e3, linewidth = 2.0, label = 'First pulse qubit A')
# plt.plot(power, T_B1*1e3, linewidth = 2.0, label = 'First pulse qubit B')
# plt.plot(power, T_A2*1e3, linestyle = '--', linewidth = 2.0, label = 'Second pulse qubit A')
# plt.plot(power, T_B2*1e3, linestyle = '--', linewidth = 2.0, label = 'Second pulse qubit B')
# plt.legend()
# plt.xlabel('Power (dBm)')
# plt.ylabel('Temperatures (mK)')

###########################################################################################
#Vary int time
f = Labber.LogFile('C:\Data\Projects\Fluxonium\Data\Augustus 18\\2019\\12\Data_1206\Histogram_correlation_v_intTime.hdf5')
# d = f.getEntry(0)
# for (channel, value) in d.items():
#     print(channel, ":", value)
signal = f.getData('AlazarTech Signal Demodulator - Channel A - Demodulated values')
intTime = f.getData('AlazarTech Signal Demodulator - Number of samples')[:13,0]
repetition = 10
corr_array = np.zeros(len(intTime ))
T_A1 = np.zeros(len(intTime ))
T_B1 = np.zeros(len(intTime ))
T_A2 = np.zeros(len(intTime ))
T_B2 = np.zeros(len(intTime ))

for idy in range(len(intTime)):
    signal_avg = []
    for idx in range(repetition):
        signal_avg = np.append(signal_avg, signal[idy*repetition+idx, :])
    herald_signal = signal_avg[0::2]
    select_signal = signal_avg[1::2]
    cov = np.corrcoef(herald_signal, select_signal)
    corr_array[idy] = abs(cov[0,1])

plt.figure(1)
plt.plot(intTime*1e-3, corr_array, linewidth = 2.0)
plt.xlabel('Integration time (ns)')
plt.ylabel('Correlation between readout (max = 1)')
plt.show()

