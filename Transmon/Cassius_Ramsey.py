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

def osc_func_decay(x,amp,freq,offset1,offset2,gamma2):
    return amp * np.cos(2 * np.pi * freq * (x - offset1))*np.exp(-(x-offset1)*gamma2) - offset2

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

def acStark_dephasing_rate(w_ac, w_c, kappa, chi):
    numerator = 8*kappa*chi**2
    denominator = (kappa**2+4*(w_ac-w_c)**2-chi**2)**2+4*chi**2*kappa**2
    return numerator*denominator**-1

def acStark_shift(w_ac, w_c, kappa, chi):
    numerator = 4*chi*(kappa**2+4*(w_ac-w_c)**2-chi**2)
    denominator = (kappa**2+4*(w_ac-w_c)-chi**2)**2 + 4*chi**2*kappa**2
    return numerator*denominator**-1

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
# path = 'G:\Projects\Fluxonium\Data\Cassius I\\2019\\09\Data_0914\Ramsey_cavity photons_3.hdf5'
# f = Labber.LogFile(path)
# d = f.getEntry(0)
# # for (channel, value) in d.items():
# #     print(channel, ":", value)
# # print ("Number of entries: ", f.getNumberOfEntries())
#
# time = f.getData('Multi-Qubit Pulse Generator - Sequence duration')[0]
# stark_freq = f.getData('R&S IQ 2 - Frequency')[:,0]
# stark_power = f.getData('R&S IQ 2 - Power')[:,0]
# signal_raw = f.getData('Signal Demodulation - Value')
# # print(signal_raw.shape)
# stark_freq = stark_freq[0:51]
# stark_power = stark_power[0::51]
# stark_power = stark_power[0:7]
# stark_power_W = 10.0**(stark_power/10.0)
# signal = np.zeros((len(time), len(stark_freq), len(stark_power)))
# gamma2_array = np.zeros((len(stark_freq), len(stark_power)))
# gamma2_err_array = np.zeros((len(stark_freq), len(stark_power)))
# freq_array = np.zeros((len(stark_freq), len(stark_power)))
# freq_err_array = np.zeros((len(stark_freq), len(stark_power)))
#
# freq_guess = 6.077e9 - 6.07533e9
# gamma2_guess = 0.5e6
#
# for idx in range(len(stark_power)):
#     for idy in range(len(stark_freq)):
#         signal_real = np.real(IQ_rotate(signal_raw[idy+51*idx,:]))
#         signal[:,idy,idx]=signal_real
#
# X,Y = np.meshgrid(time,stark_freq)
# for idx in range(len(stark_power)):
#     # plt.figure(idx)
#     # Z = signal[:,:,idx].transpose()
#     # plt.pcolormesh(X,Y,Z)
#     # print(stark_power[idx])
#     for idy in range(len(stark_freq)):
#         signal_real = signal[:, idy, idx]
#         amplitude_guess = (np.max(signal_real) - np.min(signal_real))
#         amplitude_offset = np.mean(signal_real)
#         # plt.plot(time,signal_real,'-o')
#         # try:
#         guess_list = ([-amplitude_guess, freq_guess, 0, amplitude_offset, gamma2_guess])
#         popt, pcov = curve_fit(osc_func_decay, ydata=signal_real, xdata=time, p0=guess_list)
#         perr = np.sqrt(np.diag(pcov))
#         # plt.plot(time,osc_func_decay(time,*popt))
#         # except RuntimeError:
#         #     print ('Fit error with power and freq', stark_freq[idy], stark_power[idx])
#         #     continue
#         # if perr[-1] > t2_guess:
#         #     print('Not a good fit')
#         #     continue
#         gamma2_array[idy,idx] = popt[-1]
#         gamma2_err_array[idy,idx] = perr[-1]
#         freq_array[idy,idx] = popt[1]
#         freq_err_array[idy,idx] = perr[1]
#
# gamma2 = np.mean(gamma2_array[:,0])
# freq_ramsey = np.mean(freq_array[:,0])
# plt.figure(1)
# for idx in range(2,len(stark_power)-1):
#     gamma = (gamma2_array[:,idx]-gamma2)/(stark_power_W[idx])
#     plt.errorbar(stark_freq*1e-9, gamma, linestyle='none', marker='d', mfc='none', ms=5,
#               mew=2)
#
# # plt.plot(stark_freq*1e-9,acStark_dephasing_rate(stark_freq,7.517e9, 3.5e6, 5e6)*2*np.pi)
#
# plt.figure(2)
#
# for idx in range(2,len(stark_power)-1):
#     print(stark_power[idx])
#     plt.errorbar(stark_freq*1e-9, (freq_array[:,idx]-freq_ramsey)/(stark_power_W[idx]), linestyle='none', marker='d', mfc='none', ms=5,
#               mew=2)
# plt.plot(stark_freq*1e-9,acStark_shift(stark_freq,7.517e9, 3.5e6, 5e6)*2*np.pi)

# plt.figure(3)
# for idx in range(0,len(stark_power)):
#     plt.errorbar(stark_power_W, freq_array[idx, :] -freq_ramsey,yerr = freq_err_array[idx, :], linestyle='-',
#                  marker='d', mfc='none', ms=5, mew=2, label = stark_freq[idx])
# # plt.legend()
# plt.figure(4)
# for idx in range(2,len(stark_freq)):
#     plt.errorbar(stark_power_W, gamma2_array[idx,:]-gamma2, gamma2_err_array[idx,:], linestyle='-',
#                  marker='d', mfc='none', ms=5, mew=2, label = stark_freq[idx])
# plt.legend()

#############################################################################################################
path = 'G:\Projects\Fluxonium\Data\Cassius I\\2019\\09\Data_0927\Ramsey_starkDephasing.hdf5'
f = Labber.LogFile(path)
d = f.getEntry(0)
# for (channel, value) in d.items():
#     print(channel, ":", value)
# print ("Number of entries: ", f.getNumberOfEntries())

time = f.getData('Multi-Qubit Pulse Generator - Sequence duration')[0]
stark_freq = f.getData('R&S IQ 2 - Frequency')[:,0]
stark_power = f.getData('R&S IQ 2 - Power')[:,0]
signal_raw = f.getData('Signal Demodulation - Value')
# print(signal_raw.shape)
stark_freq = stark_freq[0:71]
stark_power = stark_power[0::71]
stark_power = stark_power[0:7]
stark_power_W = 10.0**(stark_power/10.0)
signal = np.zeros((len(time), len(stark_freq), len(stark_power)))
gamma2_array = np.zeros((len(stark_freq), len(stark_power)))
gamma2_err_array = np.zeros((len(stark_freq), len(stark_power)))
freq_array = np.zeros((len(stark_freq), len(stark_power)))
freq_err_array = np.zeros((len(stark_freq), len(stark_power)))

freq_guess = 6.0765e9 - 6.07533e9
gamma2_guess = 0.5e6

for idx in range(len(stark_power)):
    for idy in range(len(stark_freq)):
        signal_real = np.real(IQ_rotate(signal_raw[idy+51*idx,:]))
        signal[:,idy,idx]=signal_real

X,Y = np.meshgrid(time,stark_freq)
for idx in range(len(stark_power)):
    # plt.figure(idx)
    # Z = signal[:,:,idx].transpose()
    # plt.pcolormesh(X,Y,Z)
    # print(stark_power[idx])
    for idy in range(len(stark_freq)):
        signal_real = signal[:, idy, idx]
        amplitude_guess = (np.max(signal_real) - np.min(signal_real))
        amplitude_offset = np.mean(signal_real)
        # plt.plot(time,signal_real,'-o')
        # try:
        guess_list = ([-amplitude_guess, freq_guess, 0, amplitude_offset, gamma2_guess])
        popt, pcov = curve_fit(osc_func_decay, ydata=signal_real, xdata=time, p0=guess_list)
        perr = np.sqrt(np.diag(pcov))
        # plt.plot(time,osc_func_decay(time,*popt))
        # except RuntimeError:
        #     print ('Fit error with power and freq', stark_freq[idy], stark_power[idx])
        #     continue
        # if perr[-1] > t2_guess:
        #     print('Not a good fit')
        #     continue
        gamma2_array[idy,idx] = popt[-1]
        gamma2_err_array[idy,idx] = perr[-1]
        freq_array[idy,idx] = popt[1]
        freq_err_array[idy,idx] = perr[1]

gamma2 = np.mean(gamma2_array[:,0])
freq_ramsey = np.mean(freq_array[:,0])
plt.figure(1)
for idx in range(2,len(stark_power)):
    gamma = (gamma2_array[:,idx]-gamma2)/(stark_power_W[idx])
    plt.errorbar(stark_freq*1e-9, gamma, linestyle='none', marker='d', mfc='none', ms=5,
              mew=2)

# plt.plot(stark_freq*1e-9,acStark_dephasing_rate(stark_freq,7.517e9, 3.5e6, 5e6)*2*np.pi)

plt.figure(2)

for idx in range(2,len(stark_power)):
    print(stark_power[idx])
    plt.errorbar(stark_freq*1e-9, (freq_array[:,idx]-freq_ramsey)/(stark_power_W[idx]), linestyle='none', marker='d', mfc='none', ms=5,
              mew=2)


plt.show()