import warnings

import h5py
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import OptimizeWarning
from scipy.optimize import curve_fit

warnings.simplefilter("error", OptimizeWarning)
warnings.simplefilter("error", RuntimeWarning)


# plt.rc('font', family='serif')

def func(x, a, b, c, d):
    return a * np.exp(-(x - c) / b) + d


def funcRamsey(x, a, b, c, d, g):
    return a * np.exp(-x / b) * np.cos(2 * np.pi * c * ((x - g))) + d


#################################################################################################################
# Parameters
directory = 'G:\Projects\Fluxonium\Data\Julius I\T2E'
fname = '081518_T1T2ET2R_loop_YOKO_94.147mA_Cav7.50923GHz_0dBm_Qubit0.5367GHz_-2dBm__PiPulse298ns_Count25_TimeStepT120000_TimeStepT2E20000_TimeStepT2R2000_loop50.h5'
path = directory + '\\' + fname
pts_num = 25
time_step_T2e = 20000
time_step_T1 = 20000
time_step_T2r = 2000
T1_guess = 100e-6
T2e_guess = 80e-6
T2r_guess = 60e-6
f_Ramsey_guess = .1e6
loop_num = 50

#################################################################################################################
time_t2e = np.linspace(0, pts_num * time_step_T2e, pts_num)
time_t1 = np.linspace(0, pts_num * time_step_T1, pts_num)
time_t2r = np.linspace(0, pts_num * time_step_T2r, pts_num)
T1_array = []
T1_err_array = []
T2e_array = []
T2e_err_array = []
T2r_array = []
T2r_err_array = []
f_R_array = []
f_R_err_array = []
Tp_array = []
loop_index = []

# Read data and fit
with h5py.File(path, 'r') as hf:
    print('List of arrays in this file: \n', hf.keys())
    count = np.array(hf.get('count'))
    phase_raw = hf.get('PHASEMAG_Phase0')
    # print phase_raw

    for idx in range(loop_num):
        phase = phase_raw[idx, 0]
        phase_t1 = phase[::3]
        phase_t2e = phase[1::3]
        phase_t2r = phase[2::3]
        phase_t1 = np.unwrap(phase_t1) * 180 / np.pi
        phase_t1 = phase_t1 - np.min(phase_t1)
        phase_t1 = abs(phase_t1)
        phase_t2e = np.unwrap(phase_t2e) * 180 / np.pi
        phase_t2e = phase_t2e - np.min(phase_t2e)
        phase_t2e = abs(phase_t2e)
        phase_t2r = np.unwrap(phase_t2r) * 180 / np.pi
        phase_t2r = phase_t2r - np.min(phase_t2r)
        phase_t2r = abs(phase_t2r)

        ###########################################################################
        guess = [phase_t1[0] - phase_t1[-1], T1_guess, 0, phase_t1[-1]]
        try:
            popt, pcov = curve_fit(func, time_t1 * 1e-9, phase_t1, guess)
        except RuntimeError:
            print("Doesn't fit well entry " + str(idx))
            continue
        except RuntimeWarning:
            print("Doesn't fit well entry " + str(idx))
            continue
        except OptimizeWarning:
            print("Doesn't fit well entry " + str(idx))
            continue
        a, b, c, d = popt  # b is T1
        time_nice = np.linspace(0, pts_num * time_step_T1, pts_num * 100)
        phase_fit = func(time_nice * 1e-9, a, b, c, d)
        perr = np.sqrt(abs(np.diag(pcov)))
        T1 = b * 1e6
        T1_err = perr[1] * 1e6
        if T1_err > T1 / 3.5:
            print("Doesn't fit well entry " + str(idx))
            continue
        # plt.figure(1)
        # plt.plot(time_t1*1e-3, phase_t1, 'k-o', alpha = 0.2)
        # plt.plot(time_nice*1e-3, phase_fit)
        # plt.xlabel('microseconds',size=12.0)
        # plt.ylabel('phase (deg)',size=12.0)
        # plt.tick_params(labelsize = 14.0)

        ############################################################################
        guess = [phase_t2e[0] - phase_t2e[-1], T2e_guess, 0, phase_t2e[-1]]
        try:
            popt, pcov = curve_fit(func, time_t2e * 1e-9, phase_t2e, guess)
        except RuntimeError:
            print("Doesn't fit well entry " + str(idx))
            continue
        except RuntimeWarning:
            print("Doesn't fit well entry " + str(idx))
            continue
        except OptimizeWarning:
            print("Doesn't fit well entry " + str(idx))
            continue
        a, b, c, d = popt  # b is T1
        time_nice = np.linspace(0, pts_num * time_step_T2e, pts_num * 100)
        phase_fit = func(time_nice * 1e-9, a, b, c, d)
        perr = np.sqrt(abs(np.diag(pcov)))
        T2e = b * 1e6
        T2e_err = perr[1] * 1e6
        if T2e_err > T2e / 2:
            print("Doesn't fit well entry " + str(idx))
            continue
        # plt.figure(2)
        # plt.plot(time_t2e*1e-3, phase_t2e, 'k-o', alpha = 0.2)
        # plt.plot(time_nice*1e-3, phase_fit)
        # plt.xlabel('microseconds',size=12.0)
        # plt.ylabel('phase (deg)',size=12.0)
        # plt.tick_params(labelsize = 14.0)

        ############################################################################

        guess = [-(np.max(phase_t2r) - np.min(phase_t2r)), T2r_guess, f_Ramsey_guess, 0, 0]
        try:
            popt, pcov = curve_fit(funcRamsey, time_t2r * 1e-9, phase_t2r, guess)
        except RuntimeError:
            print("Doesn't fit well entry " + str(idx))
            continue
        except RuntimeWarning:
            print("Doesn't fit well entry " + str(idx))
            continue
        except OptimizeWarning:
            print("Doesn't fit well entry " + str(idx))
            continue
        a, b, c, d, g = popt
        time_nice = np.linspace(0, pts_num * time_step_T2r, pts_num * 100)  # ns
        phase_fit = funcRamsey(time_nice * 1e-9, a, b, c, d, g)
        perr = np.sqrt(abs(np.diag(pcov)))
        T2r = b * 1e6
        T2r_err = perr[1] * 1e6
        f_R = c / 1e6  # MHz
        f_R_err = perr[2] / 1e6
        if T2r_err > T2r / 1.5 or T2r_err > T2e:
            print("Doesn't fit well entry " + str(idx))
            continue
        # plt.figure(3)
        # plt.plot(time_t2r*1e-3, phase_t2r, 'k-o', alpha = 0.2)
        # plt.plot(time_nice*1e-3, phase_fit)
        # plt.xlabel('microseconds',size=12.0)
        # plt.ylabel('phase (deg)',size=12.0)
        # plt.tick_params(labelsize = 14.0)

        loop_index = np.append(loop_index, idx)
        T1_array = np.append(T1_array, T1)
        T1_err_array = np.append(T1_err_array, T1_err)
        T2e_array = np.append(T2e_array, T2e)
        T2e_err_array = np.append(T2e_err_array, T2e_err)
        T2r_array = np.append(T2r_array, T2r)
        T2r_err_array = np.append(T2r_err_array, T2r_err)
        f_R_array = np.append(f_R_array, f_R)
        f_R_err_array = np.append(f_R_err_array, f_R_err)
        Tp = (T2e ** -1 - (2 * T1) ** -1) ** -1
        Tp_array = np.append(Tp_array, Tp)

plt.figure(4, figsize=[7, 5])
plt.errorbar(loop_index, T1_array, yerr=T1_err_array, fmt='s', mfc='none', mew=2.0, mec='b', ecolor='b', label=r'$T_1$')
plt.errorbar(loop_index, T2e_array, yerr=T2e_err_array, fmt='h', mfc='none', mew=2.0, mec='g', ecolor='g',
             label=r'$T_2e$')
# plt.errorbar(loop_index, T2r_array, yerr=T2r_err_array, fmt = 'D', mfc = 'none', mew = 2.0, mec = 'y', ecolor = 'y', label=r'$T_2R$')
# plt.errorbar(loop_index, Tp_array, fmt = 'd', mfc = 'none', mew = 2.0, mec = 'r', ecolor = 'r')
plt.legend()
# plt.xlabel('Index',size=12.0)
# plt.ylabel('microseconds',size=12.0)
plt.ylim([0, 180])
plt.yticks([0, 50, 100, 150])
plt.tick_params(labelsize=18.0)

# T1avg = np.average(T1_array)
# T2Eavg = np.average(T2e_array)
# T2Ravg = np.average(T2r_array)
# print('Mean T1:' + str(T1avg) + '+-' + str(np.average(T1_err_array)) + ' T2E:' + str(T2Eavg)+ '+-' + str(np.average(T2e_err_array)) + ' T2R: ' + str(T2Ravg)+ '+-' + str(np.average(T2r_err_array)))

# plt.figure(5)
# plt.errorbar(loop_index, f_R_array, yerr=f_R_err_array, fmt = 's', mfc = 'none', mew = 2.0, mec = 'r', ecolor = 'r')
# plt.xlabel('Index', size=12.0)
# plt.ylabel('Ramsey freq(MHz)',size=12.0)

# plt.figure(6)
# plt.errorbar(T1_array, T2e_array, xerr = T1_err_array, yerr=T2e_err_array, fmt='d', mfc = 'none', mew = 1.0, mec = 'm', ecolor = 'm')
# plt.xlabel('T1 (us)')
# plt.ylabel('T2 echo (us)')

# plt.figure(7)
# plt.errorbar(T1_array, T2r_array, xerr = T1_err_array, yerr=T2r_err_array, fmt='d', mfc = 'none', mew = 1.0, mec = 'orange', ecolor = 'orange')
# plt.xlabel('T1 (us)')
# plt.ylabel('T2 ramsey (us)')

plt.tick_params(labelsize=14.0)
# plt.grid()
plt.show()

print(np.corrcoef(T2e_array, T1_array)[0, 1])
