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

def osc_func(x,amp,freq,offset1,offset2):
    return amp * np.cos(2 * np.pi * freq * (x - offset1)) - offset2

def find_freq1(y_data, x_data):
    y = np.fft.fft((np.max(y_data) - np.min(y_data)) ** -1.0 * y_data)
    f = np.fft.fftfreq(len(x_data)) * (x_data[1] - x_data[0]) ** -1
    return abs(f[np.argmax(y)])