import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
plt.rc('font', family='serif')
import warnings
from scipy.optimize import OptimizeWarning
warnings.simplefilter("error", OptimizeWarning)
def T1_func(x, a, b, c, d):
     return a * np.exp(-(x - c) / b) + d

directory = 'G:\Projects\Fluxonium\Data\Fluxonium #10_python code by Jon\T1_T2_(0to1)_YOKO_38p5to38p76mA\T1_T2_(0to1)_YOKO_38p71mA'
fname = 't1_pulse_3.483e9_682'
path = directory + '\\' + fname

# t1_pulse_3.4887e9_919
# G:\Projects\Fluxonium\Data\Fluxonium #10_python code by Jon\T1_T2_(0to1)_YOKO_38p5to38p76mA\T1_T2_(0to1)_YOKO_38p71mA
#t1_pulse_3.483e9_682



err_array = []
T1_array = []
plt.figure(figsize=[7,7])
time = np.genfromtxt(path + '_time0.csv')
for idx in range(10):
    phase = np.genfromtxt(path + '_phase'+str(idx)+'.csv')
    phase = phase - np.min(phase)
    phase = -180*np.unwrap(phase)/np.pi


    guessA = np.max(phase)# - np.min(phase)
    guess = [guessA, 1e-3, 0, 0]
    try:
        popt, pcov = curve_fit(T1_func, time * 1e-9, phase, guess)
    except 'OptimizeWarning':
        continue
    except 'RuntimeError':
        continue
    a, b, c, d = popt
    err = np.sqrt(abs(np.diag(pcov)))

    T1 = b*1e3 #ms
    T1_err = err[1] * 1e3 #ms
    T1_array = np.append(T1_array, T1)
    err_array = np.append(err_array, T1_err)
    print T1
    time_nice = np.linspace(0, np.max(time), 1000)
    phase_sim = T1_func(time_nice * 1e-9, a, b, c, d)
    # plt.plot(time_nice / 1e6, phase_sim +1.3, linewidth=2.0, color = 'k')
    # plt.errorbar(time/1e3, phase - np.min(phase), fmt='d', mfc='none', mew=2.0)
    # plt.plot(time/1e6, phase +1.3, 's-', mfc='none',linewidth=0.5)
#
# plt.xlim([0,9])
# plt.yticks([0,1.2])
# plt.xticks([0,2,4,6,8])

meas_indx = np.linspace(1,len(T1_array),len(T1_array))
plt.errorbar(meas_indx, T1_array, yerr=err_array, linewidth=2.0)
plt.xlim([0.5,10.5])
plt.tick_params(labelsize = 18.0)
plt.show()