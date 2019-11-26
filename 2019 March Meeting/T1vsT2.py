import warnings

import h5py
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import OptimizeWarning
from scipy.optimize import curve_fit

warnings.simplefilter("error", OptimizeWarning)
warnings.simplefilter("error", RuntimeWarning)

plt.figure(3, figsize=[6,6])
plt.tick_params(labelsize = 18.0)

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
plt.errorbar(T1_array, T2_array, fmt = 's', mfc = 'none', mew = 2.0, label = 'Qubit A')

##########################################################################################
def func(x, a, b, c, d):
    return a*np.exp(-(x-c)/b) + d

def func_g(x, a, b, c, d):
    return a*np.exp(-((x-c)/b)**2) + d

#################################################################################################################
#Parameters
directory = 'G:\Projects\Fluxonium\Data\Fluxonium #10_7.5GHzCav\T2E'
fname = 'T1T2ET2R_loop_YOKO_28.53mA_Cav7.3649GHz_-15dBm_Qubit0.50425GHz_25dBm__PiPulse930ns_Count20_TimeStep20000.h5'
path = directory + '\\' + fname
pts_num = 20
time_step_T1 = 35000
time_step_T2E = 40000
T1_guess = 100e-6
T2_guess = 100e-6
loop_num = 20
#################################################################################################################
time_t2 = np.linspace(0, pts_num*time_step_T2E, pts_num)
time_t1 = np.linspace(0, pts_num*time_step_T1, pts_num)
T1_array = []
T1_err_array = []
T2_array = []
T2_err_array = []
Tp_array = []
loop_index = []

#Read data and fit
with h5py.File(path,'r') as hf:
    print('List of arrays in this file: \n', hf.keys())
    count = np.array(hf.get('count'))
    phase_raw = hf.get('PHASEMAG_Phase0')
    # print phase_raw

    for idx in range(loop_num):
        phase = phase_raw[idx, 0]
        phase_t1 = phase[::3]
        phase_t2 = phase[1::3]
        phase_t1 = np.unwrap(phase_t1)*180/np.pi
        phase_t1 = phase_t1 - np.min(phase_t1)
        phase_t1 = abs(phase_t1)
        phase_t2 = np.unwrap(phase_t2)*180/np.pi
        phase_t2 = phase_t2 - np.min(phase_t2)
        phase_t2 = abs(phase_t2)

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
        # plt.plot(time_t1, phase_t1, '-o', alpha = 0.2)
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
        Tp = (T2**-1 - (2*T1)**-1)**-1
        Tp_array = np.append(Tp_array, Tp)
# print len(loop_index)
# print len(T1_array)
# print len(T2_array)

plt.errorbar(T1_array, T2_array, fmt = 's', mfc = 'none', mew = 2.0, label = 'Qubit C')


def t1_func(x, a, b, c, d):
    return a*np.exp(-(x-c)/b) + d

def t2e_func(x, a, b, c, d):
    return a*np.exp(-(x-c)/b) + d


# directory = 'C:\\Data\\2019\\03\Data_0327'
# fname = 'T1_12.hdf5'
# path = directory + '\\' + fname
path = 'G:\Projects\Fluxonium\Data\\2019\\07\Data_0723\T1_T2E_41.344mA.hdf5'
t1_guess = 500e-6
t2e_guess = 500e-6
pointsnum = 41



#Read data and fit
with h5py.File(path,'r') as hf:
    print('List of arrays in this file: \n', list(hf.keys()))
    data_group = hf['Data']
    print (list(data_group.keys()))
    channel_names = data_group['Channel names']
    print (channel_names[0:])
    data = data_group['Data']
    print (data.shape)
    loop_num = int(data[0,2,-1])
    print(loop_num)
    t1_amp_array = np.zeros(loop_num)
    t1_array = np.zeros(loop_num)
    t1_error_array = np.zeros(loop_num)
    t2e_amp_array = np.zeros(loop_num)
    t2e_array = np.zeros(loop_num)
    t2e_error_array = np.zeros(loop_num)

    index = np.linspace(1, loop_num, loop_num)
    #Extract T1 Data
    # plt.figure(1)
    for idx in range(loop_num):
        time = data[0, 1, 0:pointsnum]
        time_nice = np.linspace(time[0], time[-1], 100)
        demod_real_t1 = data[0, 3, 0 + (idx * pointsnum):pointsnum + (idx * pointsnum)]
        demod_imag_t1 = data[0, 4, 0 + (idx * pointsnum):pointsnum + (idx * pointsnum)]
        demod_mag_t1 = np.absolute(demod_real_t1+1j*demod_imag_t1)
        demod_phase_t1 = np.arctan2(demod_imag_t1,demod_real_t1)*180/np.pi
        demod_phase_t1 = demod_phase_t1 - np.min(demod_phase_t1)
        demod_norm_t1 = demod_phase_t1
        # plt.plot(time*1e6, demod_norm_t1, '-d', alpha = 0.5)
        guess = ([demod_norm_t1[0]-demod_norm_t1[-1], t1_guess, 0, demod_norm_t1[-1]])
        popt, pcov = curve_fit(t2e_func, time, demod_norm_t1, guess)
        perr = np.sqrt(abs(np.diag(pcov)))
        # plt.plot(time_nice*1e6, t1_func(time_nice, *popt), linewidth = 2.0)
        # plt.xlabel('Delay time (us)')
        # plt.ylabel('')
        t1_amp_array[idx] = popt[0]
        t1_array[idx] = popt[1]*1e6
        t1_error_array[idx]=perr[1]*1e6

        # Extract T2e Data
    for idx in range(loop_num):
        time = data[0, 1, 0:pointsnum]
        time_nice = np.linspace(time[0], time[-1], 100)
        demod_real_t2e = data[1, 3, 0 + (idx * pointsnum):pointsnum+ (idx * pointsnum)]
        demod_imag_t2e = data[1, 4, 0 + (idx * pointsnum):pointsnum + (idx * pointsnum)]
        demod_mag_t2e = np.absolute(demod_real_t2e + 1j * demod_imag_t2e)
        demod_phase_t2e = np.arctan2(demod_imag_t2e, demod_real_t2e) * 180 / np.pi
        demod_phase_t2e = demod_phase_t2e - np.min(demod_phase_t2e)
        demod_norm_t2e = demod_phase_t2e
        # plt.plot(time * 1e6, demod_norm_t2e, '-d', alpha=0.5)
        guess = ([demod_norm_t2e[0] - demod_norm_t2e[-1], t2e_guess, 0, demod_norm_t2e[-1]])
        popt, pcov = curve_fit(t2e_func, time, demod_norm_t2e, guess)
        perr = np.sqrt(abs(np.diag(pcov)))
        # plt.plot(time_nice * 1e6, t2e_func(time_nice, *popt), linewidth=2.0)
        # plt.xlabel('Delay time (us)')
        # plt.ylabel('')
        t2e_amp_array[idx] = popt[0]
        t2e_array[idx] = popt[1] * 1e6
        t2e_error_array[idx] = perr[1] * 1e6

plt.errorbar(t1_array, t2e_array, fmt = 's', mfc = 'none', mew = 2.0, label = 'Qubit I')
plt.legend()
plt.yscale('log')
plt.xscale('log')
plt.ylim([50,1000])
plt.xlim([50,1000])
plt.show()
