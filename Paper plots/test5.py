# Analyze Rabi data at high T1 point, 38.71mA
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

plt.figure(figsize=(10, 10))
plt.rc('font', family='serif')


def func(x, a, b, c, d, g):
    return a * np.exp(-b * x) * np.cos(2 * np.pi * c * ((x - g))) + d


count = 0
directory = "G:\Projects\Fluxonium\Data\Fluxonium #10_python code by Jon\T1_Rabi_YOKO_38p71mA"
measurement = 'rabi_pulse_3.4887e9_910'
path = directory + '\\' + measurement
phase = np.genfromtxt(path + '_phase.csv', delimiter=',')
time = np.genfromtxt(path + '_time.csv', delimiter=',')
count = +1

phase = phase + np.genfromtxt(path + '_phase0.csv', delimiter=',')
count = +1

phase = phase + np.genfromtxt(path + '_phase1.csv', delimiter=',')
count = +1

measurement = 'rabi_pulse_3.4887e9_911'
path = directory + '\\' + measurement
phase = phase + np.genfromtxt(path + '_phase00.csv', delimiter=',')
count = +1

phase = phase + np.genfromtxt(path + '_phase01.csv', delimiter=',')
count = +1

phase = phase + np.genfromtxt(path + '_phase1.csv', delimiter=',')
count = +1

phase = phase + np.genfromtxt(path + '_phase02.csv', delimiter=',')
count = +1

phase = phase + np.genfromtxt(path + '_phase03.csv', delimiter=',')
count = +1

phase = phase / count
phase = -phase
phase = phase - np.min(phase)
phase = phase * 180 / np.pi

guessA = np.max(phase) - np.min(phase)
guess = [guessA, 0, 1e6, 0, 0]
popt, pcov = curve_fit(func, time * 1e-9, phase, guess)
a, b, c, d, g = popt
plt.plot(time / 1e3, phase, 'r.')
time_nice = np.linspace(0, np.max(time), 1000)
plt.plot(time_nice / 1e3, func(time_nice * 1e-9, a, b, c, d, g), linewidth='2', color='b')

plt.tick_params(labelsize=18)
plt.show()
