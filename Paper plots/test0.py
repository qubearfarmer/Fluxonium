# Analyze Rabi and T2 data at high T1 point, 38.6mA
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

plt.figure(figsize=(5, 4))
plt.rc('font', family='serif')

def func(x, a, b, c, d, g):
    return a * np.exp(-b * x) * np.cos(2 * np.pi * c * ((x - g))) + d

directory = "G:\Projects\Fluxonium\Data\Fluxonium #10_python code by Jon\T1_Rabi_YOKO_38p71mA"

#############################################################################################
count = 0
measurement = 'rabi_pulse_3.4887e9_912'
path = directory + '\\' + measurement
time = np.genfromtxt(path + '_time.csv', delimiter=',')
phase = np.genfromtxt(path + '_phase.csv', delimiter=',')
count =count+ 1
phase = phase * 180 / np.pi
phase = phase - np.mean(phase)
phase_all = phase
# plt.plot(time/1e3, phase, 'b-', mfc='green', mew='2', mec='red')

phase = np.genfromtxt(path + '_phase1.csv', delimiter=',')
count =count+ 1
phase = phase * 180 / np.pi
phase = phase - np.mean(phase)
phase_all = phase_all + phase
# plt.plot(time/1e3, phase, 'r-', mfc='green', mew='2', mec='red')

phase = np.genfromtxt(path + '_phase2.csv', delimiter=',')
count =count+ 1
phase = phase * 180 / np.pi
phase = phase - np.mean(phase)
phase_all = phase_all + phase
# plt.plot(time/1e3, phase, 'g-', mfc='green', mew='2', mec='red')

phase = np.genfromtxt(path + '_phase3.csv', delimiter=',')
# count =count+ 1
phase = phase * 180 / np.pi
phase = phase - np.mean(phase)
# phase_all = phase_all + phase
# plt.plot(time/1e3, phase, 'y-', mfc='green', mew='2', mec='red')

phase = np.genfromtxt(path + '_phase4.csv', delimiter=',')
count = count + 1
phase = phase * 180 / np.pi
phase = phase - np.mean(phase)
phase_all = phase_all + phase
# plt.plot(time/1e3, phase, 'm-', mfc='green', mew='2', mec='red')

##################################################################
# Rabi
phase = phase_all / count
phase = - phase
phase = phase - np.mean(phase)
plt.plot(time/1e3, phase, 'd', mfc='none', mew=2.0, mec='red')
guessA = np.max(phase) - np.min(phase)
guess = [guessA, 0, 1e6, 0, 0]
popt, pcov = curve_fit(func, time * 1e-9, phase, guess)
a, b, c, d, g = popt
print ('Rabi freq ='+str(c/1e6) +'MHz')
print ('Pi pulse ='+str(1/c*1e9/2) + 'ns')
time_nice = np.linspace(0, np.max(time), 1000)
plt.plot(time_nice / 1e3, func(time_nice * 1e-9, a, b, c, d, g), linewidth=2.0, color='k')

#T2 exp
def T2_echo_func(x,a,b,c,d):
    return a*np.exp(-(x-c)/b) + d
#T2 gaussian
# def T2_echo_func(x,a,b,c,d):
#     return a*np.exp(-(x-c)**2.0/b**2.0) + d
#T2 combo
# def T2_echo_func(x,a,b,c,d,e,f):
#     return (a*np.exp(-(x-c)**2.0/b**2.0))*(np.exp(-(x-d)/e))+f
measurement = 't2_echo_pulse_3.4887e9_930'
path = directory + '\\' + measurement
time = np.genfromtxt(path + '_time.csv', delimiter=',')*2
phase = np.genfromtxt(path + '_phase.csv', delimiter=',')
phase = phase * 180 / np.pi
phase = - phase
phase = phase - np.min(phase)
phase_all = phase
count=1
for idx in range(0,6):
    phase = np.genfromtxt(path + '_phase' + str(idx) + '.csv', delimiter=',')
    phase = phase * 180 / np.pi
    phase = - phase
    phase = phase - np.min(phase)
    phase_all = phase_all + phase
    count = count + 1
    # plt.plot(time / 1e3, phase, '.')
    # plt.title(str(idx))
    # plt.show()

phase = phase_all / count
phase = phase - np.min(phase)
guessA = np.max(phase)-np.min(phase)
guess = [guessA, 4e-6, 0, 0]

popt,pcov = curve_fit(T2_echo_func, time*1e-9, phase, guess)
time_nice = np.linspace(0, np.max(time), 1000)
t2_fit = T2_echo_func(time_nice * 1e-9, *popt)
offset = t2_fit[-1]
plt.plot(time/1e3,phase - offset, 'p', mfc='none', mew=2.0, mec='g')
plt.plot(time_nice / 1e3, t2_fit - offset, linewidth=2.0, color='k')
print ('T2_echo='+str(popt[1]*1e6)+'us')
plt.tick_params(labelsize=18)
plt.xlim([0,10])
plt.ylim([-0.8,0.8])
plt.yticks(np.linspace(-0.8,0.8,5))
plt.xticks(np.linspace(0,10,5))

####################################################################################################################################
##T1
# def T1_func(x, a, b, c, d):
#     return a * np.exp(-(x - c) / b) + d
#
#
# measurement = 't1_pulse_3.4887e9_919'
# path = directory + '\\' + measurement
# count = 0
# phase_all = np.zeros(40)
# for idx in range(1, 15):
#     time = np.genfromtxt(path + '_time' + str(idx) + '.csv', delimiter=',')
#     phase = np.genfromtxt(path + '_phase' + str(idx) + '.csv', delimiter=',')
#     phase = phase * 180 / np.pi
#     phase = - phase
#     phase = phase - np.min(phase)
#     if idx in [1, 2, 3, 4, 5, 6]:
#         continue
#     phase_all = phase_all + phase
#     count = count + 1
#
#     # plt.plot(time / 1e3, phase, '.')
#     # plt.title(str(idx))
#     # plt.show()
#
# phase = phase_all / count
# phase = phase - np.min(phase)
# guessA = np.max(phase) - np.min(phase)
# guess = [guessA, 1e-3, 0, 0]
# popt, pcov = curve_fit(T1_func, time * 1e-9, phase, guess)
# a, b, c, d = popt
# print a
# time_nice = np.linspace(0, np.max(time), 1000)
# plt.plot(time / 1e3, phase - guessA / 2, 's', mfc='none', mew=2.0, mec='b')
# plt.plot(time_nice / 1e3, T1_func(time_nice * 1e-9, a, b, c, d) - guessA / 2, linewidth=2.0, color='k')
# print 'T1=' + str(b * 1e6) + 'us'
# plt.tick_params(labelsize=20)
# plt.xlim([0, 5000])
# plt.xticks(np.linspace(0,5000,3))
# plt.ylim([-0.8, 0.8])
# plt.yticks([])


####################################################################################################################################
plt.show()