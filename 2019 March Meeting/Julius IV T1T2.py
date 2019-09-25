import numpy as np
import h5py
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit


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
    plt.figure(1)
    for idx in range(loop_num):
        time = data[0, 1, 0:pointsnum]
        time_nice = np.linspace(time[0], time[-1], 100)
        demod_real_t1 = data[0, 3, 0 + (idx * pointsnum):pointsnum + (idx * pointsnum)]
        demod_imag_t1 = data[0, 4, 0 + (idx * pointsnum):pointsnum + (idx * pointsnum)]
        demod_mag_t1 = np.absolute(demod_real_t1+1j*demod_imag_t1)
        demod_phase_t1 = np.arctan2(demod_imag_t1,demod_real_t1)*180/np.pi
        demod_phase_t1 = demod_phase_t1 - np.min(demod_phase_t1)
        demod_norm_t1 = demod_phase_t1
        plt.plot(time*1e6, demod_norm_t1, '-d', alpha = 0.5)
        guess = ([demod_norm_t1[0]-demod_norm_t1[-1], t1_guess, 0, demod_norm_t1[-1]])
        popt, pcov = curve_fit(t2e_func, time, demod_norm_t1, guess)
        perr = np.sqrt(abs(np.diag(pcov)))
        plt.plot(time_nice*1e6, t1_func(time_nice, *popt), linewidth = 2.0)
        plt.xlabel('Delay time (us)')
        plt.ylabel('')
        t1_amp_array[idx] = popt[0]
        t1_array[idx] = popt[1]*1e6
        t1_error_array[idx]=perr[1]*1e6

        # Extract T2e Data
    plt.figure(2)
    for idx in range(loop_num):
        time = data[0, 1, 0:pointsnum]
        time_nice = np.linspace(time[0], time[-1], 100)
        demod_real_t2e = data[1, 3, 0 + (idx * pointsnum):pointsnum+ (idx * pointsnum)]
        demod_imag_t2e = data[1, 4, 0 + (idx * pointsnum):pointsnum + (idx * pointsnum)]
        demod_mag_t2e = np.absolute(demod_real_t2e + 1j * demod_imag_t2e)
        demod_phase_t2e = np.arctan2(demod_imag_t2e, demod_real_t2e) * 180 / np.pi
        demod_phase_t2e = demod_phase_t2e - np.min(demod_phase_t2e)
        demod_norm_t2e = demod_phase_t2e
        plt.plot(time * 1e6, demod_norm_t2e, '-d', alpha=0.5)
        guess = ([demod_norm_t2e[0] - demod_norm_t2e[-1], t2e_guess, 0, demod_norm_t2e[-1]])
        popt, pcov = curve_fit(t2e_func, time, demod_norm_t2e, guess)
        perr = np.sqrt(abs(np.diag(pcov)))
        plt.plot(time_nice * 1e6, t2e_func(time_nice, *popt), linewidth=2.0)
        plt.xlabel('Delay time (us)')
        plt.ylabel('')
        t2e_amp_array[idx] = popt[0]
        t2e_array[idx] = popt[1] * 1e6
        t2e_error_array[idx] = perr[1] * 1e6

    plt.figure(3)
    plt.figure(figsize=[7, 2.5])
    plt.errorbar(index, t1_array, yerr=t1_error_array, linestyle='none', marker='d', mfc='none', ecolor = 'blue', mec='blue', ms=5,
                 mew=2, label='$T_1$')
    plt.errorbar(index, t2e_array, yerr=t2e_error_array, linestyle='none', marker='d', mfc='none', ecolor = 'green', mec='green', ms=5,
                 mew=2, label='$T_2$')

    # plt.xlabel('Index')
    # plt.ylabel('T1, T2e (us)')
    plt.tick_params(labelsize = 18.0)
    plt.ylim(0,800)
    # plt.legend(fontsize = 16.0)
    # plt.figure(4)
    # plt.errorbar(t1_array, t2e_array)

plt.show()