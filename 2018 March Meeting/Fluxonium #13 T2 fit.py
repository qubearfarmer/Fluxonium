import h5py
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

plt.figure(figsize=(3.5, 3.5))
plt.rc('font', family='san-serif')


def func(x,a,b,c,d):
    return a*np.exp(-(x-c)**2.0/b**2.0) + d


directory = 'G:\Projects\Fluxonium\Data\Fluxonium #13\T2E'
'''
measurement = '090717_T2E_YOKO_61.04mA_Cav7.36923GHz_-30dBm_Qubit0.992GHz_25dBm_PiPulse211ns_Count30_TimeStep800_Avg_10000.h5'
path = directory + '\\' + measurement
pts_num = 30
time_step = 1000
time = np.linspace(0, pts_num*time_step, pts_num)
t2_guess = 12e-6
#Read data and fit
with h5py.File(path,'r') as hf:
    print('List of arrays in this file: \n', hf.keys())
    count = np.array(hf.get('count'))
    phase_raw = hf.get('PHASEMAG_Phase0')
    # print phase_raw
    phase = phase_raw[0, 0]
    phase = np.unwrap(phase)*180/np.pi
    phase = phase - np.min(phase)
    phase = abs(phase[0:])
plt.errorbar(time*1e-3, phase,fmt='h', mfc='none', mew=2.0, mec='green')
# plt.errorbar(qp_element[:, 0] ** 2 + qp_element[:, 1] ** 2, T1_final, yerr=T1_err_final, fmt='s', mfc='none', mew=2.0, mec='blue')

guess = [phase[0]-phase[-1], t2_guess, 0, phase[-1]]
popt, pcov = curve_fit(func, time*1e-9, phase, guess)
a,b,c,d = popt #b is T1
time_nice  = np.linspace(0, pts_num*time_step, pts_num*100)
phase_fit = func(time_nice*1e-9, a, b, c, d)
perr = np.sqrt(abs(np.diag(pcov)))
plt.plot(time_nice*1e-3, phase_fit, color='black')
plt.xticks([0,15,30])
plt.yticks([0,2,4])
plt.xlim([0,30])
plt.ylim([-0.2,4.2])

'''
def func(x,a,b,c,d):
    return a*np.exp(-(x-c)/b) + d
measurement = '090517_T2E_YOKO_61.115mA_Cav7.36923GHz_-30dBm_Qubit0.7895GHz_25dBm_PiPulse268ns_Count30_TimeStep10000_Avg_10000.h5'
path = directory + '\\' + measurement
pts_num = 30
time_step = 10000
time = np.linspace(0, pts_num*time_step, pts_num)
t2_guess = 100e-6
#Read data and fit
with h5py.File(path,'r') as hf:
    print('List of arrays in this file: \n', hf.keys())
    count = np.array(hf.get('count'))
    phase_raw = hf.get('PHASEMAG_Phase0')
    # print phase_raw
    phase = phase_raw[0, 0]
    phase = np.unwrap(phase)*180/np.pi
    phase = phase - np.min(phase)
    phase = abs(phase[0:])
plt.errorbar(time*1e-3, phase,fmt='h', mfc='none', mew=2.0, mec='green')
# plt.errorbar(qp_element[:, 0] ** 2 + qp_element[:, 1] ** 2, T1_final, yerr=T1_err_final, fmt='s', mfc='none', mew=2.0, mec='blue')

guess = [phase[0]-phase[-1], t2_guess, 0, phase[-1]]
popt, pcov = curve_fit(func, time*1e-9, phase, guess)
a,b,c,d = popt #b is T1
time_nice  = np.linspace(0, pts_num*time_step, pts_num*100)
phase_fit = func(time_nice*1e-9, a, b, c, d)
perr = np.sqrt(abs(np.diag(pcov)))
plt.plot(time_nice*1e-3, phase_fit, color='black')
plt.yticks([0,1,2])
plt.xticks([0,150,300])
# plt.title(str(b*1e6)+ r'$\pm$' +str(perr[1]*1e6))

#'''
plt.tick_params(labelsize = 18.0)
plt.show()