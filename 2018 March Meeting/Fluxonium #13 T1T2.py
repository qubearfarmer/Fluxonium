import warnings

import h5py
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import OptimizeWarning
from scipy.optimize import curve_fit

warnings.simplefilter("error", OptimizeWarning)
warnings.simplefilter("error", RuntimeWarning)

def func(x, a, b, c, d):
    return a*np.exp(-(x-c)/b) + d

def func_g(x, a, b, c, d):
    return a*np.exp(-((x-c)/b)**2) + d

#################################################################################################################
#Parameters
directory = 'G:\Projects\Fluxonium\Data\Fluxonium #13\T2E'
fname = 'T1T2ELoop_YOKO_91.015mA_Cav7.3692GHz_-30dBm_Qubit0.7821GHz_25dBm_PiPulse264ns_Count30_TimeStepT2E10000_TimeStepT120000.h5'
path = directory + '\\' + fname
pts_num = 30
time_step_T1 = 20000
time_step_T2E = 10000
T1_guess = 100e-6
T2_guess = 100e-6
loop_num = 51
#################################################################################################################
time_t2 = np.linspace(0, pts_num*time_step_T2E, pts_num)
time_t1 = np.linspace(0, pts_num*time_step_T1, pts_num)
T1_array = []
T1_err_array = []
T2_array = []
T2_err_array = []
Tp_array = []
loop_index = []
phase_t1_avg = 0
phase_t2_avg = 0
#Read data and fit
with h5py.File(path,'r') as hf:
    print('List of arrays in this file: \n', hf.keys())
    count = np.array(hf.get('count'))
    phase_raw = hf.get('PHASEMAG_Phase0')
    # print phase_raw

    for idx in range(loop_num):
        phase = phase_raw[idx, 0]
        phase_t1 = phase[::2]
        phase_t2 = phase[1::2]
        phase_t1 = np.unwrap(phase_t1)*180/np.pi
        phase_t1 = phase_t1 - np.min(phase_t1)
        phase_t1 = abs(phase_t1)
        phase_t2 = np.unwrap(phase_t2)*180/np.pi
        phase_t2 = phase_t2 - np.min(phase_t2)
        phase_t2 = abs(phase_t2)
        phase_t1_avg = phase_t1_avg + phase_t1
        phase_t2_avg = phase_t2_avg + phase_t2
        guess = [phase_t1[0]-phase_t1[-1], T1_guess, 0, phase_t1[-1]]
        try:
            popt, pcov = curve_fit(func, time_t1*1e-9, phase_t1, guess)
        except RuntimeError:
            print ("Doesn't fit well entry " + str(idx))
            continue
        except RuntimeWarning:
            print ("Doesn't fit well entry " + str(idx))
            continue
        except OptimizeWarning:
            print ("Doesn't fit well entry " + str(idx))
            continue
        a,b,c,d = popt #b is T1
        time_nice  = np.linspace(0, pts_num*time_step_T1, pts_num*100)
        phase_fit = func(time_nice*1e-9, a, b, c, d)
        perr = np.sqrt(abs(np.diag(pcov)))
        T1 = b*1e6
        T1_err = perr[1]*1e6
        # plt.figure(1)
        # plt.plot(time_t1, phase_t1, 'k-o', alpha = 0.2)
        # plt.plot(time_nice, phase_fit)
        ############################################################################

        guess = [phase_t2[0]-phase_t2[-1], T2_guess, 0, phase_t2[-1]]
        try:
            popt, pcov = curve_fit(func, time_t2*1e-9, phase_t2, guess)
        except RuntimeError:
            print ("Doesn't fit well entry " + str(idx))
            continue
        except RuntimeWarning:
            print ("Doesn't fit well entry " + str(idx))
            continue
        except OptimizeWarning:
            print ("Doesn't fit well entry " + str(idx))
            continue
        a,b,c,d = popt #b is T1
        time_nice  = np.linspace(0, pts_num*time_step_T2E, pts_num*100)
        phase_fit = func(time_nice*1e-9, a, b, c, d)
        perr = np.sqrt(abs(np.diag(pcov)))
        T2 = b*1e6
        T2_err = perr[1]*1e6
        # plt.figure(2)
        # plt.plot(time_t2, phase_t2, 'k-o', alpha = 0.2)
        # plt.plot(time_nice, phase_fit)
        loop_index = np.append(loop_index, idx)

        T1_array = np.append(T1_array, T1)
        T1_err_array = np.append(T1_err_array, T1_err)
        T2_array = np.append(T2_array, T2)
        T2_err_array = np.append(T2_err_array, T2_err)
        # Tp = (T2**-1 - (2*T1)**-1)**-1
        # Tp_array = np.append(Tp_array, Tp)
# print len(loop_index)
# print len(T1_array)
# print len(T2_array)
plt.figure(3, figsize=[7,2.5])
plt.errorbar(loop_index, T1_array, yerr=T1_err_array, fmt = 's', mfc = 'none', mew = 2.0, mec = 'b', ecolor = 'b', label=r'$T_1$')
# plt.errorbar(T1_array, T2_array, xerr=T1_err_array, yerr=T2_err_array, fmt = 's', mfc = 'none', mew = 2.0, mec = 'g', ecolor = 'g')
# plt.errorbar(T1_array, (T2_array**-1-(T1_array*2)**-1)**-1, xerr=T1_err_array, yerr=T2_err_array, fmt = 's', mfc = 'none', mew = 2.0, mec = 'g', ecolor = 'g')
plt.errorbar(loop_index, T2_array, yerr=T2_err_array, fmt = 'h', mfc = 'none', mew = 2.0, mec = 'g', ecolor = 'g', label=r'$T_2$')
# plt.errorbar(loop_index, (T2_array**-1 - (2*T1_array)**-1)**-1 , yerr=T2_err_array, fmt = 'h', mfc = 'none', mew = 2.0, mec = 'r', ecolor = 'r')
# plt.yscale("log")

plt.tick_params(labelsize = 16)
plt.yticks([0,50,100,150,200])
plt.ylim([0,200])
plt.xlim([-0.5,50.5])
plt.xticks([0, 10, 20, 30, 40,50])
#plt.errorbar(loop_index, Tp_array, fmt = 'd', mfc = 'none', mew = 2.0, mec = 'r', ecolor = 'r')
# plt.xlabel('Index')
# plt.ylabel(r'$\mu s$')
# plt.grid()

# x = np.linspace(1,1000,1001)
# plt.plot(x,2*x)
plt.legend()
path_save = "C:\\Users\\nguyen89\\Desktop\\Paper figures 2019_11_24\Fig4ctop.pdf"
plt.savefig(path_save,dpi=300,format='pdf')
plt.show()

# print (np.corrcoef(T2_array, T1_array)[0,1])
plt.figure(4)
phase_t1_avg = phase_t1_avg/loop_num
guess = [phase_t1_avg[0]-phase_t1_avg[-1], T1_guess, 0, phase_t1_avg[-1]]
popt, pcov = curve_fit(func, time_t1*1e-9, phase_t1_avg, guess)
a,b,c,d = popt #b is T1
time_nice  = np.linspace(0, pts_num*time_step_T1, pts_num*100)
phase_fit = func(time_nice*1e-9, a,b,c,d)
perr = np.sqrt(abs(np.diag(pcov)))
T1 = b*1e6
T1_err = perr[1]*1e6
plt.plot(time_t1/1e3, phase_t1_avg, 'bs', linewidth = 2)
plt.plot(time_nice/1e3, phase_fit)
plt.yscale('log')
print (T1, T1_err)
plt.tick_params(labelsize = 18)

plt.figure(5)
phase_t2_avg = phase_t2_avg/loop_num
guess = [phase_t2_avg[0]-phase_t2_avg[-1], T2_guess, 0, phase_t2_avg[-1]]
popt, pcov = curve_fit(func, time_t2*1e-9, phase_t2_avg, guess)
a,b,c,d = popt #b is T1
time_nice  = np.linspace(0, pts_num*time_step_T2E, pts_num*100)
phase_fit = func(time_nice*1e-9, a, b, c, d)
perr = np.sqrt(abs(np.diag(pcov)))
T2 = b*1e6
T2_err = perr[1]*1e6
plt.plot(time_t2/1e3, phase_t2_avg, 'gd')
plt.plot(time_nice/1e3,phase_fit, linewidth = 2)
print (T2, T2_err)
plt.tick_params(labelsize = 18)

plt.plot()
