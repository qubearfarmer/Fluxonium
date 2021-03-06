import numpy as np
from matplotlib import pyplot as plt
import sys
from scipy.optimize import curve_fit
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

def randomized_benchmarking_0(x,p,a,b):
    return a*p**x+b

def randomized_benchmarking_1(x,p,a,b,c):
    return a*p**x++b+c*(x-1)*p**(x-2)
#############################################################################################
#Hero RB
# path = 'G:\Projects\Fluxonium\Data\Cassius I\\2019\\09\Data_0916\Randomized_benchmarking.hdf5'
# f = Labber.LogFile(path)
# d = f.getEntry(0)
# for (channel, value) in d.items():
#     print(channel, ":", value)
# print ("Number of entries: ", f.getNumberOfEntries())
#
# cliff_num = f.getData('Multi-Qubit Pulse Generator - Number of Cliffords')[0]
# randomize = f.getData('Multi-Qubit Pulse Generator - Randomize')[:,0]
# signal = f.getData('Signal Demodulation - Value')
# signal = np.average(signal,axis = 0)
# signal = IQ_rotate(signal)
# signal_real = np.real(signal)
# # signal_real = signal_real - np.min(signal_real)
# plt.plot(cliff_num, signal_real*1e6,'d')
# plt.tick_params(labelsize = 14.0)
# plt.ylabel('I signal (uV)', size = 14.0)
# plt.xlabel('# randomized pulses', size = 14.0)

#Fit
# n=1
# d = 2**n
# guess =([1,np.max(signal_real)-np.min(signal_real),np.min(signal_real)])
# opt,cov = curve_fit(randomized_benchmarking_0,ydata = signal_real, xdata = cliff_num, p0 = guess)
# err = np.sqrt(np.diag(cov))
# parameter = opt[0]
# parameter_err = err[0]
# error = (d-1)*(1-parameter)/d
# error_err = (d-1)*parameter_err/d
# error = error/1.875
# error_err = error_err/1.875
# print('0-order model fidelity',(1-error))
# print('0-order model fidelity error',(error_err))
# plt.plot(cliff_num, randomized_benchmarking_0(cliff_num,*opt)*1e6)
#
# guess =([1,np.max(signal_real)-np.min(signal_real),np.min(signal_real),0])
# opt,cov = curve_fit(randomized_benchmarking_1,ydata = signal_real, xdata = cliff_num, p0 = guess)
# err = np.sqrt(np.diag(cov))
# parameter = opt[0]
# parameter_err = err[0]
# error = (d-1)*(1-parameter)/d
# error_err = (d-1)*parameter_err/d
# error = error/1.875
# error_err = error_err/1.875
# print('1-order model fidelity',(1-error))
# print('1-order model fidelity error',(error_err))
# plt.plot(cliff_num, randomized_benchmarking_1(cliff_num,*opt)*1e6)

################################################################################
#Sweeping DRAG coefficient parameters
path = 'G:\Projects\Fluxonium\Data\Cassius I\\2019\\09\Data_0916\Randomized_benchmarking_Out.hdf5'
f = Labber.LogFile(path)
d = f.getEntry(0)
for (channel, value) in d.items():
    print(channel, ":", value)
print ("Number of entries: ", f.getNumberOfEntries())

cliff_num = f.getData('Multi-Qubit Pulse Generator - Number of Cliffords')[0]
randomize = f.getData('Multi-Qubit Pulse Generator - Randomize')[:10,0]
drag_scaling = f.getData('Multi-Qubit Pulse Generator - DRAG scaling #1')[0::10,0]
signal = f.getData('Signal Demodulation - Value')
fidelity0 = np.zeros(len(drag_scaling))
fidelity0_error = np.zeros(len(drag_scaling))
fidelity1 = np.zeros(len(drag_scaling))
fidelity1_error = np.zeros(len(drag_scaling))
signal_average = np.zeros([len(drag_scaling),len(cliff_num)])
#slice and average the signal for every drag scaling
for drag_idx in range(len(drag_scaling)):
    signal_average[drag_idx,:] = np.average(signal[drag_idx*len(randomize):(drag_idx+1)*len(randomize),:],axis=0)
    signal_average[drag_idx,:] = IQ_rotate(signal_average[drag_idx,:])
    signal_real = np.real(signal_average[drag_idx,:])
    # plt.plot(cliff_num, signal_real)

    n=1
    d = 2**n
    guess =([1,np.max(signal_real)-np.min(signal_real),np.min(signal_real)])
    opt,cov = curve_fit(randomized_benchmarking_0,ydata = signal_real, xdata = cliff_num, p0 = guess)
    err = np.sqrt(np.diag(cov))
    parameter = opt[0]
    parameter_err = err[0]
    error = (d-1)*(1-parameter)/d
    error_err = (d-1)*parameter_err/d
    error = error/1.875
    error_err = error_err/1.875
    fidelity0 [drag_idx] = 1 - error
    fidelity0_error [drag_idx] = error_err

    # guess = ([1.0, np.max(signal_real) - np.min(signal_real), np.min(signal_real), 0])
    # try:
    #     opt,cov = curve_fit(randomized_benchmarking_1,ydata = signal_real, xdata = cliff_num, p0 = guess)
    # except RuntimeError:
    #     continue
    # err = np.sqrt(np.diag(cov))
    # parameter = opt[0]
    # parameter_err = err[0]
    # error = (d-1)*(1-parameter)/d
    # error_err = (d-1)*parameter_err/d
    # error = error/1.875
    # error_err = error_err/1.875
    # fidelity1[drag_idx] = 1 - error
    # fidelity1_error[drag_idx] = error_err

plt.errorbar(drag_scaling*1e12, fidelity0, yerr=fidelity0_error, linestyle='none', marker='d', mfc='none', ms=5,
              mew=2)
# plt.errorbar(drag_scaling*1e12, fidelity1, yerr=fidelity1_error, linestyle='none', marker='d', mfc='none', ms=5,
#               mew=2)
plt.tick_params(labelsize = 14.0)
plt.ylabel('Fidelity', size = 14.0)
plt.xlabel('DRAG scaling (ps)', size = 14.0)
plt.show()