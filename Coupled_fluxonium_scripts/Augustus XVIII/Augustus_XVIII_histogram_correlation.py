import numpy as np
import sys
sys.path.append('C:\Program Files (x86)\Labber\Script')
import Labber
from matplotlib import pyplot as plt
from qutip import*
from scipy.optimize import curve_fit

#constants
kB = 1.38e-23
h = 6.626e-34

#Read file
f = Labber.LogFile('Z:\Projects\Fluxonium\Data\Augustus 18\\2020\\01\Data_0116\Histogram_correlation.hdf5')
# guess2D = np.array([4350, -800, 7150, -500, 13000, -200, 14000, 200, 20])
# f = Labber.LogFile('C:\Data\Projects\Fluxonium\Data\Augustus 18\\2019\\11\Data_1120\Histogram_correlation_3.hdf5')
# guess2D = np.array([4350, -800, 7150, -500, 13000, -200, 14000, 200, 20])

#Define functions
def gaussian (x,a,x0,sigma):
    return a*np.exp(-(x-x0)**2/sigma**2)
def gaussian4(x,a1,x1,a2,x2,a3,x3,a4,x4,sigma):
    return a1*np.exp(-(x-x1)**2/sigma**2) + a2*np.exp(-(x-x2)**2/sigma**2) + \
           a3*np.exp(-(x-x3)**2/sigma**2) + a4*np.exp(-(x-x4)**2/sigma**2)
def gaussian2d(coord,a,x0,y0,sigma):
    return a*np.exp((-(coord[0]-x0)**2-(coord[1]-y0)**2)/sigma**2)


signal = f.getData('AlazarTech Signal Demodulator - Channel A - Demodulated values')*1e6
# delay_time = f.getData('Multi-Qubit Pulse Generator - Delay after heralding')[:,0]
repetition = 1
signal_avg = []
for idx in range(repetition):
    signal_avg = np.append(signal_avg, signal[idx, :])
herald_signal = signal_avg[0::2]
select_signal = signal_avg[1::2]
cov = np.corrcoef(herald_signal, select_signal)
print (abs(cov[0,1]))

######################################################################
# Temperatures extract and compare, 4 blobs
sReal = np.real(herald_signal)
sImag = np.imag(herald_signal)
#
plt.figure(1)
H, xedges, yedges = np.histogram2d(sReal,sImag, bins = [100,100])
H = H.T
X,Y = np.meshgrid(xedges, yedges)
plt.pcolormesh(X,Y,H, cmap = 'GnBu')
#
plt.figure(2)
counts, edges = np.histogram(sReal, bins = 100)
plt.plot(edges[:-1], counts, label = 'First pulse')
# guess = np.array([550, -2800, 13150, -1800, 23000, -700, 6000, 100, 20])
# opt,cov = curve_fit(gaussian4, edges[:-1],counts, guess)
# axis_nice = np.linspace(edges[0], edges[-1], 1001)
# plt.plot(axis_nice, gaussian4(axis_nice,*opt))
# a1,x1,a2,x2,a3,x3,a4,x4,sigma = opt
# plt.plot(axis_nice, gaussian(axis_nice,a1,x1,sigma))
# plt.plot(axis_nice, gaussian(axis_nice,a2,x2,sigma))
# plt.plot(axis_nice, gaussian(axis_nice,a3,x3,sigma))
# plt.plot(axis_nice, gaussian(axis_nice,a4,x4,sigma))
# pgg = a4/(a1+a2+a3+a4)
# peg = a3/(a1+a2+a3+a4)
# pge = a1/(a1+a2+a3+a4)
# pee = a2/(a1+a2+a3+a4)
# f_a = 72.4e6
# f_b = 136.3e6
# T_a = h*f_a/(-kB*np.log((pee+peg)/(pgg+pge)))
# T_b = h*f_b/(-kB*np.log((pee+pge)/(pgg+peg)))
# populations = dict({'P_gg=': round(pgg,3), 'P_eg=': round(peg,3), 'P_ge=': round(pge,3), 'P_ee=': round(pee,3)})
# temperatures = dict({'T_A(mK)=': round(T_a*1e3,3), 'T_B(mK)=':round(T_b*1e3,3)})
# print (temperatures)
# textstr =""
# for key in temperatures:
#     textstr = textstr+key+str(temperatures[key])+'\n'
# plt.text(-1000,1000,textstr)
#
# #Second pulse
# sReal = np.real(select_signal)
# sImag = np.imag(select_signal)
# plt.figure(3)
#
# H, xedges, yedges = np.histogram2d(sReal,sImag, bins = [100,100])
# H = H.T
# X,Y = np.meshgrid(xedges, yedges)
# plt.pcolormesh(X,Y,H, cmap = 'GnBu')
# #
# plt.figure(2)
# counts, edges = np.histogram(sReal, bins = 100)
# plt.plot(edges[:-1], counts, label = 'Second pulse')
# #
# opt,cov = curve_fit(gaussian4, edges[:-1],counts, guess)
# axis_nice = np.linspace(edges[0], edges[-1], 1001)
# plt.plot(axis_nice, gaussian4(axis_nice,*opt))
# a1,x1,a2,x2,a3,x3,a4,x4,sigma = opt
# pgg = a3/(a1+a2+a3+a4)
# peg = a4/(a1+a2+a3+a4)
# pge = a2/(a1+a2+a3+a4)
# pee = a1/(a1+a2+a3+a4)
# f_a = 72.4e6
# f_b = 136.3e6
# T_a = h*f_a/(-kB*np.log((pee+peg)/(pgg+pge)))
# T_b = h*f_b/(-kB*np.log((pee+pge)/(pgg+peg)))
# populations = dict({'P_gg=': round(pgg,3), 'P_eg=': round(peg,3), 'P_ge=': round(pge,3), 'P_ee=': round(pee,3)})
# temperatures = dict({'T_A(mK)=': round(T_a*1e3,3), 'T_B(mK)=':round(T_b*1e3,3)})
# print (temperatures)
#
# ##########################################################################################
# #Pre-seection
# selected_signal = []
# preselected_signal = []
# xmin = -200
# xmax = 400
# ymin = -1700
# ymax = -1100
#
# # xmax = 250e-6
# # xmin = 50e-6
# # ymax = -350e-6
# # ymin = -550e-6
#
# # xmax = -400e-6
# # xmin = -600e-6
# # ymax = 200e-6
# # ymin = -50e-6
#
# # xmax = -650e-6
# # xmin = -850e-6
# # ymax = -100e-6
# # ymin = -300e-6
# for idx in range(len(herald_signal)):
#     if (np.real(herald_signal[idx]) > xmin) and (np.real(herald_signal[idx]) < xmax) \
#             and (np.imag(herald_signal[idx]) > ymin) and (np.imag(herald_signal[idx]) < ymax):
#         preselected_signal = np.append(preselected_signal, select_signal[idx])
#         selected_signal = np.append(selected_signal, herald_signal[idx])
#
# sReal = np.real(preselected_signal)
# sImag = np.imag(preselected_signal)
# plt.figure(4)
# H, xedges, yedges = np.histogram2d(sReal,sImag, bins = [100,100])
# H = H.T
# X,Y = np.meshgrid(xedges, yedges)
# plt.pcolormesh(X,Y,H, cmap = 'GnBu')
#
# plt.figure(5)
# counts, edges = np.histogram(sReal, bins = 100)
# plt.plot(edges[:-1], counts, label = 'Preselected signal')
# guess = np.array([0, -2800, 13150, -1800, 0, -700, 0, 100, 20])
# opt,cov = curve_fit(gaussian4, edges[:-1],counts, guess)
# axis_nice = np.linspace(edges[0], edges[-1], 1001)
# plt.plot(axis_nice, gaussian4(axis_nice,*opt), label = 'fit')
# a1,x1,a2,x2,a3,x3,a4,x4,sigma = opt
# pgg = a4/(a1+a2+a3+a4)
# peg = a3/(a1+a2+a3+a4)
# pge = a1/(a1+a2+a3+a4)
# pee = a2/(a1+a2+a3+a4)
# f_a = 72.5e6
# f_b = 136.5e6
# T_a = h*f_a/(-kB*np.log((pee+peg)/(pgg+pge)))
# T_b = h*f_b/(-kB*np.log((pee+pge)/(pgg+peg)))
# populations = dict({'P_gg=': round(pgg,3), 'P_eg=': round(peg,3), 'P_ge=': round(pge,3), 'P_ee=': round(pee,3)})
# temperatures = dict({'T_A(mK)=': round(T_a*1e3,3), 'T_B(mK)=':round(T_b*1e3,3)})
# print (populations)
# # sReal = np.real(selected_signal)*1e6
# # counts, edges = np.histogram(sReal, bins = 100)
# # plt.plot(edges[:-1], counts, label = 'Ideal signal')
# plt.legend()
# plt.show()
