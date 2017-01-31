# All Rabi amplitude
import warnings

import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import OptimizeWarning
from scipy.optimize import curve_fit

warnings.simplefilter('error', OptimizeWarning)
plt.figure(figsize=(10, 10))
plt.rc('font', family='serif')


def T1_func(x, a, b, c, d):
    return a * np.exp(-(x - c) / b) + d


directory = "G:\Projects\Fluxonium\Data\Fluxonium #10_python code by Jon\T1_Rabi_YOKO_38p71mA"
measurement = 't1_pulse_3.486e9_944'
path = directory + '\\' + measurement
time = np.genfromtxt(path + '_time0' + '.csv')
# print time
T1_array = []
T1_error_array = []
idx_array = []
# '''
for idx in range(0, 426):
    phase = np.genfromtxt(path + '_phase' + str(idx) + '.csv')
    phase = phase * 180 / np.pi
    phase = - phase
    phase = phase - np.min(phase)
    guessA = np.max(phase) - np.min(phase)
    guess = [guessA, 1e-3, 0, 0]
    if guessA < 0.2 or guessA > 1.2:
        continue
    try:
        popt, pcov = curve_fit(T1_func, time * 1e-9, phase, guess)
    except RuntimeError:
        continue
    except OptimizeWarning:
        continue
    a, b, c, d = popt
    perr = np.sqrt(np.diag(pcov))
    if b * 1e6 < 500 or b * 1e6 > 5000 or perr[1] * 1e6 > 500:
        continue
        # if b*1e6 > 3000:
        # print str(idx)
        # print 'Amplitide = ' +str(a)
        # print 'T1 = ' + str(b*1e6) + 'us'
    plt.plot(time / 1e3, phase, time / 1e3, T1_func(time * 1e-9, a, b, c, d))
    plt.title(str(b * 1e6) + ', ' + str(perr[1] * 1e6))
    T1_array = np.append(T1_array, b * 1e6)
    T1_error_array = np.append(T1_error_array, b * 1e6, perr[1] * 1e6)
    idx_array = np.append(idx_array, idx)
    plt.show()
'''
for idx in range(np.int(426/10)):
    count = 0
    for i in range(10):
        phase = np.genfromtxt(path + '_phase' + str(idx*20+i) + '.csv')
        phase = phase * 180 / np.pi
        phase = - phase
        phase = phase - np.min(phase)
        guessA = np.max(phase) - np.min(phase)
        if guessA < 0.2 or guessA > 1.2:
            continue
        if i==0:
            phase_sum = phase
        else:
            phase_sum = phase_sum + phase
            count = count+1
    phase_avg = phase_sum/count
    guessA = np.max(phase_avg) - np.min(phase_avg)
    guess = [guessA, 1e-3, 0, 0]
    if guessA < 0.2 or guessA > 1.2:
        continue
    try:
        popt, pcov = curve_fit(T1_func, time * 1e-9, phase, guess)
    except RuntimeError:
        continue
    except OptimizeWarning:
        continue
    a, b, c, d = popt
    perr = np.sqrt(np.diag(pcov))
    if b * 1e6 < 500 or b * 1e6 > 5000:
    #     plt.plot(time/1e3, phase_avg)
    #     plt.show()
        continue
    T1_error = perr[1]
    plt.semilogy(time/1e3, phase_avg, 'rd', time/1e3, T1_func(time*1e-9,a,b,c,d), 'b-')
    plt.title('T1='+str(b*1e6)+'us, error='+str(T1_error*1e6)+'us')
    T1_array = np.append(T1_array, b*1e6)
    plt.show()
# '''
# plt.plot(T1_array,'md')
# plt.ylim([500,4500])
# y,x = np.histogram(T1_array, np.linspace(500,4500, 10))
# # print x,y
# plt.hist(T1_array, bins = np.linspace(500,4500, 9))
# plt.tick_params(labelsize=16)
# plt.show()
