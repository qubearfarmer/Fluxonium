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
path = 'Z:\Projects\Fluxonium\Data\Augustus 18\\2019\\12\Data_1227\RB_AWG_qubitB_2.hdf5'
f = Labber.LogFile(path)
# d = f.getEntry(0)
# for (channel, value) in d.items():
#     print(channel, ":", value)
# print ("Number of entries: ", f.getNumberOfEntries())
#
cliff_num = f.getData('Multi-Qubit Pulse Generator - Number of Cliffords')[0]
# randomize = f.getData('Multi-Qubit Pulse Generator - Randomize')[:,0]
signal = f.getData('AlazarTech Signal Demodulator - Channel A - Average demodulated value')*1e6
# print (signal.shape)
signal = np.average(signal,axis = 0)
# print (signal.shape)
signal = IQ_rotate(signal)
signal_real = np.real(signal)
signal_imag = np.imag(signal)
signal_mag = abs(signal)
signal_phase = np.unwrap(np.angle(signal))
toFit = signal_real
toFit = toFit - np.min(toFit)
plt.plot(cliff_num, toFit,'d')
plt.tick_params(labelsize = 14.0)
plt.ylabel('I signal (uV)', size = 14.0)
plt.xlabel('# randomized pulses', size = 14.0)

#Fit
n=1
d = 2**n
guess =([1,np.max(toFit)-np.min(toFit),np.min(toFit)])
opt,cov = curve_fit(randomized_benchmarking_0, ydata = toFit, xdata = cliff_num, p0 = guess)
err = (np.sqrt(np.diag(cov)))
parameter = (opt[0])
parameter_err = err[0]
error = abs((d-1)*(1-parameter)/d)
error_err = (d-1)*parameter_err/d
error = error/1.875
error_err = error_err/1.875
print('0-order model fidelity',(1-error))
print('0-order model fidelity error',(error_err))
plt.plot(cliff_num, randomized_benchmarking_0(cliff_num,*opt), label = 'Zeroth order fit')
# #
# guess =([0.99,np.max(signal_real)-np.min(signal_real),np.min(signal_real),0])
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
# plt.plot(cliff_num, randomized_benchmarking_1(cliff_num,*opt), label = 'First order fit')
# plt.legend()
plt.show()